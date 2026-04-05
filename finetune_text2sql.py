#!/usr/bin/env python3
"""

数据（两种格式）：
1) alpaca：build_cot_ft_dataset.py to-alpaca 生成的 JSON 数组
   [{ "instruction", "input", "output" }, ...]
2) sharegpt：Qwen/ShareGPT 风格 JSONL（本项目的 `qwen3_finetune_sharegpt.jsonl`）
   {"messages":[{"role":"system|user|assistant","content":"..."} , ...]}

示例：
  cd nlp && source .venv/bin/activate
  pip install -r requirements_ft.txt
  python finetune_text2sql.py \\
    --base_model Qwen/Qwen3-4B-Instruct-2507 \\
    --data_path data/spider_alpaca_524.json \\
    --data_format alpaca \\
    --output_dir trained_models/qwen3-4b-spider-cot-lora

  python finetune_text2sql.py \\
    --base_model Qwen/Qwen3-4B-Instruct-2507 \\
    --data_path qwen3_finetune_sharegpt.jsonl \\
    --data_format sharegpt \\
    --output_dir trained_models/qwen3-4b-sharegpt-cot-lora
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

_ROOT = Path(__file__).resolve().parent


def _alpaca_prompt(instruction: str, input_text: str, output: str) -> str:
    if (input_text or "").strip():
        return (
            "Below is an instruction that describes a task, paired with an input that provides "
            "further context. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )
    return (
        "Below is an instruction that describes a task. Write a response that appropriately "
        "completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{output}"
    )


def _infer_data_format(path: Path, explicit: str) -> str:
    if explicit and explicit != "auto":
        return explicit
    # Heuristic: jsonl => sharegpt; json => alpaca array
    return "sharegpt" if path.suffix.lower() == ".jsonl" else "alpaca"


def main() -> None:
    p = argparse.ArgumentParser(description="LoRA finetune for Text2SQL (nlp-local)")
    p.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--data_path", type=str, default="data/spider_alpaca_524.json")
    p.add_argument(
        "--data_format",
        type=str,
        default="auto",
        choices=["auto", "alpaca", "sharegpt"],
        help="auto: infer from suffix; alpaca: instruction/input/output array; sharegpt: jsonl with messages",
    )
    p.add_argument("--output_dir", type=str, default="trained_models/qwen3-4b-spider-cot-lora")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument(
        "--cutoff_len",
        type=int,
        default=3072,
        help="max sequence length; lower if OOM (eval needs large logits buffer)",
    )
    p.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="keep 1 for long-seq causal LM eval to avoid OOM",
    )
    p.add_argument("--val_set_size", type=int, default=52)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="comma-separated",
    )
    p.add_argument("--train_on_inputs", action="store_true", help="train loss on full sequence")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.chdir(_ROOT)
    data_path = Path(args.data_path)
    if not data_path.is_file():
        print(f"ERROR: data not found: {data_path.resolve()}", file=sys.stderr)
        sys.exit(2)

    use_gc = args.gradient_checkpointing and not args.no_gradient_checkpointing
    target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]

    torch.manual_seed(args.seed)
    gradient_accumulation_steps = max(1, args.batch_size // args.micro_batch_size)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    hf_config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    if getattr(hf_config, "quantization_config", None) is not None:
        hf_config.quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=hf_config,
        dtype=torch.float16,
        device_map={"": local_rank},
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def tokenize_one(text: str, add_eos: bool = True):
        out = tokenizer(
            text,
            truncation=True,
            max_length=args.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        ids = out["input_ids"]
        if add_eos and ids and ids[-1] != tokenizer.eos_token_id and len(ids) < args.cutoff_len:
            ids = ids + [tokenizer.eos_token_id]
            out["attention_mask"] = out["attention_mask"] + [1]
        out["input_ids"] = ids
        out["labels"] = list(ids)
        return out

    data_format = _infer_data_format(data_path, args.data_format)

    def _tokenize_alpaca(batch):
        full = _alpaca_prompt(batch["instruction"], batch.get("input") or "", batch["output"])
        tok_full = tokenize_one(full, add_eos=True)
        if not args.train_on_inputs:
            user_only = _alpaca_prompt(batch["instruction"], batch.get("input") or "", "")
            tok_user = tokenize_one(user_only, add_eos=False)
            ulen = len(tok_user["input_ids"])
            tok_full["labels"] = [-100] * ulen + tok_full["labels"][ulen:]
        return tok_full

    def _tokenize_sharegpt(batch):
        messages = batch.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("sharegpt format requires a non-empty 'messages' list")
        if args.train_on_inputs:
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return tokenize_one(full_text, add_eos=True)

        prompt_msgs = messages[:-1]
        last = messages[-1]
        assistant_text = last.get("content", "") if isinstance(last, dict) else ""
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + (assistant_text or "")
        tok_full = tokenize_one(full_text, add_eos=True)
        tok_prompt = tokenize_one(prompt_text, add_eos=False)
        ulen = len(tok_prompt["input_ids"])
        tok_full["labels"] = [-100] * ulen + tok_full["labels"][ulen:]
        return tok_full

    def generate_and_tokenize(batch):
        if data_format == "alpaca":
            return _tokenize_alpaca(batch)
        if data_format == "sharegpt":
            return _tokenize_sharegpt(batch)
        raise ValueError(f"unknown data_format: {data_format}")

    # FP16 LoRA：peft>=0.17 无 prepare_model_for_int8_training；checkpointing 由 TrainingArguments 打开。
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    if use_gc:
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    if data_format == "sharegpt":
        train_only = load_dataset("json", data_files=str(data_path), split="train")
        data = {"train": train_only}
    else:
        data = load_dataset("json", data_files=str(data_path))
    if args.val_set_size > 0:
        split = data["train"].train_test_split(
            test_size=min(args.val_set_size, len(data["train"]) - 1),
            shuffle=True,
            seed=args.seed,
        )
        train_ds = split["train"].shuffle(seed=args.seed).map(
            generate_and_tokenize, remove_columns=split["train"].column_names
        )
        eval_ds = split["test"].shuffle(seed=args.seed).map(
            generate_and_tokenize, remove_columns=split["test"].column_names
        )
    else:
        train_ds = data["train"].shuffle(seed=args.seed).map(
            generate_and_tokenize, remove_columns=data["train"].column_names
        )
        eval_ds = None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        # 默认 False 时验证常会升到 FP32，logits[seq,vocab] 极大，易在第一步 eval OOM
        fp16_full_eval=True,
        gradient_checkpointing=use_gc,
        logging_steps=10,
        optim="adamw_torch",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
    )
    if eval_ds is not None:
        targs.eval_strategy = "steps"
        targs.eval_steps = args.eval_steps
        targs.load_best_model_at_end = True
        targs.metric_for_best_model = "eval_loss"
        targs.greater_is_better = False

    model.config.use_cache = False
    # 不要 patch state_dict：新版 PEFT + Trainer 下会导致 checkpoint / save_pretrained 写出空 adapter。

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    trainer.train()
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Done. LoRA saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
