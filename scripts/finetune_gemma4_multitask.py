from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from womenhelp_competition.hf_utils import configure_hf_backend  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning LoRA de Gemma 4 para WomenHelp")
    parser.add_argument("--train-jsonl", default=str(REPO_ROOT / "artifacts" / "sft_data" / "joint_aligned_train.jsonl"))
    parser.add_argument("--eval-jsonl", default=str(REPO_ROOT / "artifacts" / "sft_data" / "joint_aligned_devel.jsonl"))
    parser.add_argument("--exclude-tasks", default="")
    parser.add_argument("--model-name-or-path", default=os.environ.get("GEMMA4_MODEL_NAME_OR_PATH", "unsloth/gemma-4-E2B-it"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "finetune" / "gemma4_joint_aligned_lora"))
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="fp32")
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--gemma4-empty-thought-channel", action="store_true")
    return parser.parse_args()


def inject_gemma4_empty_thought_channel(text: str) -> str:
    marker = "<|turn>model\n"
    empty_channel = "<|channel>thought\n<channel|>\n"
    if marker not in text or "<|channel>thought" in text:
        return text
    return text.replace(marker, marker + empty_channel)


def formatting_prompts_func(chat_template_source, inject_empty_thought_channel: bool):
    def _formatter(examples):
        convos = examples.get("conversations", examples.get("messages"))
        texts = []
        for convo in convos:
            text = chat_template_source.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
            )
            if inject_empty_thought_channel:
                text = inject_gemma4_empty_thought_channel(text)
            texts.append(text.removeprefix("<bos>"))
        return {"text": texts}

    return _formatter


def unwrap_text_tokenizer(tokenizer):
    text_tokenizer = tokenizer
    seen_ids = set()
    while hasattr(text_tokenizer, "tokenizer"):
        current_id = id(text_tokenizer)
        if current_id in seen_ids:
            break
        seen_ids.add(current_id)
        next_tokenizer = getattr(text_tokenizer, "tokenizer")
        if next_tokenizer is text_tokenizer:
            break
        text_tokenizer = next_tokenizer
    return text_tokenizer


def load_chat_tokenizer(model_name_or_path: str):
    processor = None
    tokenizer_error = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    except Exception as exc:  # pragma: no cover - fallback path depends on remote model metadata
        tokenizer_error = exc
        tokenizer = None

    if tokenizer is None:
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        tokenizer = unwrap_text_tokenizer(processor)

    chat_template_source = tokenizer if hasattr(tokenizer, "apply_chat_template") else processor
    if chat_template_source is None or not hasattr(chat_template_source, "apply_chat_template"):
        if tokenizer_error is not None:
            raise RuntimeError(
                f"No se pudo cargar un tokenizer/chat template compatible para {model_name_or_path}"
            ) from tokenizer_error
        raise RuntimeError(
            f"No se pudo cargar un tokenizer/chat template compatible para {model_name_or_path}"
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer, chat_template_source


def resolve_lora_target_modules(model, target_modules_arg: str):
    if target_modules_arg.strip() == "all-linear":
        return "all-linear"

    requested_suffixes = {
        module_name.strip()
        for module_name in target_modules_arg.split(",")
        if module_name.strip()
    }
    resolved_modules = []
    for module_name, module in model.named_modules():
        if module_name.rsplit(".", 1)[-1] not in requested_suffixes:
            continue
        if isinstance(module, torch.nn.Linear):
            resolved_modules.append(module_name)

    if not resolved_modules:
        requested = ", ".join(sorted(requested_suffixes))
        raise RuntimeError(f"No se resolvieron módulos LoRA compatibles para: {requested}")
    return sorted(set(resolved_modules))


def tokenize_texts(tokenizer, max_seq_length: int):
    def _tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            add_special_tokens=False,
        )

    return _tokenize


def parse_excluded_tasks(exclude_tasks_arg: str) -> set[str]:
    return {
        task_name.strip()
        for task_name in exclude_tasks_arg.split(",")
        if task_name.strip()
    }


def main() -> None:
    args = parse_args()
    configure_hf_backend()

    effective_logging_steps = max(1, min(args.logging_steps, args.max_steps))
    effective_save_steps = max(1, min(args.save_steps, args.max_steps))
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    load_dtype = dtype_map[args.precision]
    use_bf16 = args.precision == "bf16"
    use_fp16 = args.precision == "fp16"

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = load_dataset("json", data_files=str(args.train_jsonl), split="train")
    eval_dataset = None
    if args.eval_jsonl and Path(args.eval_jsonl).exists():
        eval_dataset = load_dataset("json", data_files=str(args.eval_jsonl), split="train")

    excluded_tasks = parse_excluded_tasks(args.exclude_tasks)
    if excluded_tasks:
        train_dataset = train_dataset.filter(lambda example: example.get("task") not in excluded_tasks)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.filter(lambda example: example.get("task") not in excluded_tasks)

    print(f"train rows crudos: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"eval rows crudos: {len(eval_dataset)}")
    if excluded_tasks:
        print(f"tareas excluidas: {sorted(excluded_tasks)}")

    train_dataset = train_dataset.filter(lambda example: example.get("messages") is not None)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.filter(lambda example: example.get("messages") is not None)

    print(f"train rows estandarizados: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"eval rows estandarizados: {len(eval_dataset)}")
    print(f"logging steps efectivos: {effective_logging_steps}")
    print(f"save steps efectivos: {effective_save_steps}")
    print(f"precision: {args.precision}")
    print(f"max grad norm: {args.max_grad_norm}")
    print(f"gradient checkpointing: {args.gradient_checkpointing}")
    print(f"gemma4 empty thought channel: {args.gemma4_empty_thought_channel}")
    print(f"train jsonl: {args.train_jsonl}")
    if eval_dataset is not None:
        print(f"eval jsonl: {args.eval_jsonl}")
    print("backend: transformers + peft")

    tokenizer, chat_template_source = load_chat_tokenizer(args.model_name_or_path)

    # La resolución de módulos LoRA queda por sufijo para soportar Gemma y Qwen
    # sin acoplar el script a un árbol interno específico del modelo.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=load_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    formatter = formatting_prompts_func(chat_template_source, args.gemma4_empty_thought_channel)
    train_columns = train_dataset.column_names
    train_dataset = train_dataset.map(formatter, batched=True, remove_columns=train_columns)
    if eval_dataset is not None:
        eval_columns = eval_dataset.column_names
        eval_dataset = eval_dataset.map(formatter, batched=True, remove_columns=eval_columns)

    tokenize_fn = tokenize_texts(tokenizer, args.max_seq_length)
    train_text_columns = train_dataset.column_names
    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=train_text_columns)
    if eval_dataset is not None:
        eval_text_columns = eval_dataset.column_names
        eval_dataset = eval_dataset.map(tokenize_fn, batched=True, remove_columns=eval_text_columns)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_target_modules = resolve_lora_target_modules(model, args.lora_target_modules)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        target_modules=lora_target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    if isinstance(lora_target_modules, list):
        print(f"módulos LoRA resueltos: {len(lora_target_modules)}")
    else:
        print(f"módulos LoRA resueltos: {lora_target_modules}")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        args=TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=args.gradient_checkpointing,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            logging_steps=effective_logging_steps,
            save_steps=effective_save_steps,
            save_total_limit=args.save_total_limit,
            bf16=use_bf16,
            fp16=use_fp16,
            optim=args.optim,
            report_to="none",
            remove_unused_columns=False,
            seed=args.seed,
        ),
    )
    train_result = trainer.train()

    trainer.model.save_pretrained(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))

    metrics = dict(train_result.metrics)
    metrics["backend"] = "transformers_peft"
    metrics["excluded_tasks"] = sorted(excluded_tasks)
    metrics["precision"] = args.precision
    metrics["optim"] = args.optim
    metrics["learning_rate"] = args.learning_rate
    metrics["max_grad_norm"] = args.max_grad_norm
    metrics["weight_decay"] = args.weight_decay
    metrics["max_seq_length"] = args.max_seq_length
    metrics["gemma4_empty_thought_channel"] = args.gemma4_empty_thought_channel
    metrics["lora_target_module_count"] = len(lora_target_modules) if isinstance(lora_target_modules, list) else None
    metrics["lora_target_module_suffixes"] = (
        sorted({module_name.rsplit(".", 1)[-1] for module_name in lora_target_modules})
        if isinstance(lora_target_modules, list)
        else [lora_target_modules]
    )
    metrics["train_rows"] = len(train_dataset)
    if eval_dataset is not None:
        metrics["eval_rows"] = len(eval_dataset)
    with (output_dir / "train_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    with (output_dir / "train_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
