from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch  # noqa: E402

from womenhelp_competition.data import (  # noqa: E402
    label_names_from_vector,
    load_subtask1,
    load_subtask2,
    slice_records,
)
from womenhelp_competition.hf_utils import configure_hf_backend  # noqa: E402
from womenhelp_competition.labels import (  # noqa: E402
    SUBTASK1_ID_TO_NAME,
    SUBTASK1_LABELS,
    SUBTASK1_NAME_TO_ID,
    SUBTASK2_COLUMNS,
    SUBTASK2_LABELS,
)
from womenhelp_competition.metrics import (  # noqa: E402
    compute_multiclass_metrics,
    compute_multilabel_metrics,
)
from womenhelp_competition.parsing import extract_json_candidate, parse_joint_prediction  # noqa: E402
from womenhelp_competition.prompts import (  # noqa: E402
    BASE_SYSTEM_PROMPT,
    COMPACT_SYSTEM_PROMPT,
    EVIDENCE_JSON_SYSTEM_PROMPT,
    EXPERT_SYSTEM_PROMPT,
    build_joint_prompt,
    build_joint_prompt_compact,
    build_joint_prompt_evidence_json,
    build_joint_prompt_expert,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Corre baseline zero-shot de Gemma 4")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--record-ids-path", default=None)
    parser.add_argument("--model-name-or-path", default=os.environ.get("GEMMA4_MODEL_NAME_OR_PATH", "unsloth/gemma-4-E2B-it"))
    parser.add_argument(
        "--inference-backend",
        choices=["unsloth", "transformers"],
        default=os.environ.get("GEMMA4_INFERENCE_BACKEND", "unsloth"),
    )
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "prompt_baseline"))
    parser.add_argument("--subtask1-split", default="devel")
    parser.add_argument("--subtask2-split", default="devel")
    parser.add_argument("--limit-subtask1", type=int, default=0)
    parser.add_argument("--limit-subtask2", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--min-new-tokens", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--max-consecutive-blank-batches", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--renormalize-logits", action="store_true")
    parser.add_argument("--suppress-pad-token", action="store_true")
    parser.add_argument("--fallback-min-new-tokens", type=int, default=0)
    parser.add_argument("--fallback-temperature", type=float, default=-1.0)
    parser.add_argument("--fallback-top-p", type=float, default=-1.0)
    parser.add_argument("--severity-fallback", default="High", choices=SUBTASK1_LABELS)
    parser.add_argument("--prompt-style", choices=["simple", "expert", "compact", "evidence_json"], default="simple")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--prompt-format",
        choices=["auto", "chat", "plain", "literal_chat"],
        default="auto",
    )
    parser.add_argument(
        "--transformers-dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "eager", "sdpa"],
        default="auto",
    )
    parser.add_argument("--debug-trace-limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_processed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    processed = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                processed.add(json.loads(line)["record_id"])
    return processed


def load_record_ids(path: str | None) -> set[str] | None:
    if not path:
        return None
    record_ids = {
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return record_ids or None


def filter_records_by_id(records: list[object], selected_ids: set[str] | None) -> list[object]:
    if selected_ids is None:
        return records
    return [record for record in records if record.record_id in selected_ids]


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def batched_records(records: list[object], batch_size: int) -> list[list[object]]:
    return [records[index : index + batch_size] for index in range(0, len(records), batch_size)]


def should_report_progress(completed: int, total: int, report_every: int, batch_size: int) -> bool:
    if total <= 10 or completed >= total:
        return True
    previous_completed = max(completed - batch_size, 0)
    return (previous_completed // report_every) != (completed // report_every)


def unwrap_text_tokenizer(tokenizer_or_processor):
    current = tokenizer_or_processor
    seen_ids: set[int] = set()
    while hasattr(current, "tokenizer"):
        current_id = id(current)
        if current_id in seen_ids:
            break
        seen_ids.add(current_id)
        next_tokenizer = getattr(current, "tokenizer")
        if next_tokenizer is current:
            break
        current = next_tokenizer
    return current


def resolve_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return None
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map[dtype_name]


def select_chat_template_source(processor, tokenizer):
    # Algunos modelos multimodales exponen el chat template en el processor y otros
    # solo en el tokenizer interno; aquí resolvemos ambas variantes sin asumir una sola API.
    if processor is not None and hasattr(processor, "apply_chat_template"):
        if getattr(processor, "chat_template", None) not in {None, ""}:
            return processor
        wrapped_tokenizer = getattr(processor, "tokenizer", None)
        if wrapped_tokenizer is not None and getattr(wrapped_tokenizer, "chat_template", None) not in {None, ""}:
            return processor
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        if getattr(tokenizer, "chat_template", None) not in {None, ""}:
            return tokenizer
    return None


def load_unsloth_stack(model_name_or_path: str, max_seq_length: int):
    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name_or_path,
        dtype=None,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        full_finetuning=False,
    )
    model.for_inference()
    chat_template_source = select_chat_template_source(None, tokenizer)
    return model, tokenizer, chat_template_source


def load_transformers_stack(
    model_name_or_path: str,
    transformers_dtype: str,
    attn_implementation: str,
):
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    processor = None
    tokenizer = None
    try:
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        tokenizer = unwrap_text_tokenizer(processor)
    except Exception:
        processor = None

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    torch_dtype = resolve_torch_dtype(transformers_dtype)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if attn_implementation != "auto":
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model = model.to("cuda")
    model.eval()

    chat_template_source = select_chat_template_source(processor, tokenizer)
    return model, tokenizer, chat_template_source


def load_generation_stack(args: argparse.Namespace):
    if args.inference_backend == "unsloth":
        return load_unsloth_stack(args.model_name_or_path, args.max_seq_length)
    return load_transformers_stack(
        args.model_name_or_path,
        args.transformers_dtype,
        args.attn_implementation,
    )


def resolve_prompt_format(chat_template_source, prompt_format: str) -> str:
    supports_chat_template = chat_template_source is not None
    if prompt_format == "auto":
        return "chat" if supports_chat_template else "plain"
    if prompt_format == "chat" and not supports_chat_template:
        raise ValueError("--prompt-format=chat requiere un tokenizer/processor con chat template")
    return prompt_format


def render_chat_prompt(chat_template_source, messages: list[dict[str, str]], enable_thinking: bool) -> str:
    try:
        prompt_text = chat_template_source.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        if enable_thinking:
            raise ValueError("El chat template actual no soporta enable_thinking=True")
        prompt_text = chat_template_source.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return inject_gemma4_empty_thought_channel(prompt_text) if not enable_thinking else prompt_text


def inject_gemma4_empty_thought_channel(text: str) -> str:
    marker = "<|turn>model\n"
    empty_channel = "<|channel>thought\n<channel|>\n"
    if marker not in text or "<|channel>thought" in text:
        return text
    return text.replace(marker, marker + empty_channel)


def build_plain_prompt_text(system_prompt: str, user_prompt: str) -> str:
    return (
        "Instrucciones:\n"
        f"{system_prompt.strip()}\n\n"
        "Tarea:\n"
        f"{user_prompt.strip()}\n\n"
        "Respuesta JSON:\n"
    )


def build_literal_chat_prompt_text(system_prompt: str, user_prompt: str) -> str:
    return inject_gemma4_empty_thought_channel(
        "<bos><|turn>system\n"
        f"{system_prompt.strip()}<turn|>\n"
        "<|turn>user\n"
        f"{user_prompt.strip()}<turn|>\n"
        "<|turn>model\n"
    )



def build_prompt_text(
    chat_template_source,
    text: str,
    prompt_style: str,
    prompt_format: str,
    enable_thinking: bool,
) -> str:
    system_prompt = BASE_SYSTEM_PROMPT
    user_prompt = build_joint_prompt(text)
    if prompt_style == "expert":
        system_prompt = EXPERT_SYSTEM_PROMPT
        user_prompt = build_joint_prompt_expert(text)
    elif prompt_style == "compact":
        system_prompt = COMPACT_SYSTEM_PROMPT
        user_prompt = build_joint_prompt_compact(text)
    elif prompt_style == "evidence_json":
        system_prompt = EVIDENCE_JSON_SYSTEM_PROMPT
        user_prompt = build_joint_prompt_evidence_json(text)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if prompt_format == "chat":
        if chat_template_source is None:
            raise ValueError("No hay chat template disponible para renderizar el prompt")
        return render_chat_prompt(chat_template_source, messages, enable_thinking)
    if prompt_format == "plain":
        return build_plain_prompt_text(system_prompt, user_prompt)
    if prompt_format == "literal_chat":
        return build_literal_chat_prompt_text(system_prompt, user_prompt)
    raise ValueError(f"prompt_format no soportado: {prompt_format}")


def build_generate_kwargs(
    tokenizer,
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    renormalize_logits: bool,
    suppress_pad_token: bool,
) -> dict[str, object]:
    generate_kwargs: dict[str, object] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "do_sample": temperature > 0,
    }
    if min_new_tokens > 0:
        generate_kwargs["min_new_tokens"] = min_new_tokens
    if temperature > 0:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
        if top_k > 0:
            generate_kwargs["top_k"] = top_k
    if repetition_penalty != 1.0:
        generate_kwargs["repetition_penalty"] = repetition_penalty
    if no_repeat_ngram_size > 0:
        generate_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    if renormalize_logits:
        generate_kwargs["renormalize_logits"] = True
    if suppress_pad_token and tokenizer.pad_token_id is not None and tokenizer.pad_token_id != tokenizer.eos_token_id:
        generate_kwargs["bad_words_ids"] = [[tokenizer.pad_token_id]]
    return generate_kwargs


def generate_once_batch(
    model,
    tokenizer,
    chat_template_source,
    texts: list[str],
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    renormalize_logits: bool,
    suppress_pad_token: bool,
    prompt_style: str,
    prompt_format: str,
    enable_thinking: bool,
    origin_batch_size: int | None = None,
    fallback_depth: int = 0,
    fallback_mode: str = "primary_batch",
) -> list[dict[str, object]]:
    if not texts:
        return []

    prompt_texts = [
        build_prompt_text(chat_template_source, text, prompt_style, prompt_format, enable_thinking)
        for text in texts
    ]
    tokens = tokenizer(text=prompt_texts, return_tensors="pt", padding=True, add_special_tokens=False)
    prompt_length = tokens["input_ids"].shape[1]
    input_token_counts = tokens["attention_mask"].sum(dim=1).tolist()
    tokens = {key: value.to("cuda") for key, value in tokens.items()}
    generate_kwargs = build_generate_kwargs(
        tokenizer,
        max_new_tokens,
        min_new_tokens,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        no_repeat_ngram_size,
        renormalize_logits,
        suppress_pad_token,
    )

    with torch.inference_mode():
        outputs = model.generate(**tokens, **generate_kwargs)

    results: list[dict[str, object]] = []
    resolved_origin_batch_size = origin_batch_size or len(texts)
    for prompt_text, input_token_count, row in zip(prompt_texts, input_token_counts, outputs):
        generated_ids = row[prompt_length:]
        raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        raw_output_with_special_tokens = tokenizer.decode(generated_ids, skip_special_tokens=False)
        results.append(
            {
                "prompt_text": prompt_text,
                "raw_output": raw_output,
                "raw_output_with_special_tokens": raw_output_with_special_tokens,
                "input_token_count": int(input_token_count),
                "origin_batch_size": resolved_origin_batch_size,
                "effective_batch_size": len(texts),
                "fallback_depth": fallback_depth,
                "fallback_mode": fallback_mode,
                "enable_thinking": enable_thinking,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "no_repeat_ngram_size": no_repeat_ngram_size,
                "renormalize_logits": renormalize_logits,
                "suppress_pad_token": suppress_pad_token,
            }
        )
    return results


def generate_responses_batch(
    model,
    tokenizer,
    chat_template_source,
    texts: list[str],
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    renormalize_logits: bool,
    suppress_pad_token: bool,
    fallback_min_new_tokens: int,
    fallback_temperature: float,
    fallback_top_p: float,
    prompt_style: str,
    prompt_format: str,
    enable_thinking: bool,
    origin_batch_size: int | None = None,
    fallback_depth: int = 0,
) -> list[dict[str, object]]:
    resolved_origin_batch_size = origin_batch_size or len(texts)
    responses = generate_once_batch(
        model,
        tokenizer,
        chat_template_source,
        texts,
        max_new_tokens,
        min_new_tokens,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        no_repeat_ngram_size,
        renormalize_logits,
        suppress_pad_token,
        prompt_style,
        prompt_format,
        enable_thinking,
        origin_batch_size=resolved_origin_batch_size,
        fallback_depth=fallback_depth,
        fallback_mode="primary_batch" if fallback_depth == 0 else "recursive_batch",
    )

    blank_indices = [index for index, response in enumerate(responses) if not response["raw_output"]]
    if not blank_indices:
        return responses

    retry_min_new_tokens = fallback_min_new_tokens if fallback_min_new_tokens >= 0 else min_new_tokens
    retry_temperature = fallback_temperature if fallback_temperature >= 0 else temperature
    retry_top_p = fallback_top_p if fallback_top_p >= 0 else top_p
    print(
        f"Lote con {len(blank_indices)}/{len(texts)} salidas vacías; "
        f"reintentando en fallback (profundidad={fallback_depth + 1})."
    )

    if len(texts) == 1:
        # Si solo falló un ejemplo, hacemos el reintento aislado para distinguir
        # entre un problema del prompt puntual y un colapso del lote completo.
        retry_response = generate_once_batch(
            model,
            tokenizer,
            chat_template_source,
            texts,
            max_new_tokens,
            retry_min_new_tokens,
            retry_temperature,
            retry_top_p,
            top_k,
            repetition_penalty,
            no_repeat_ngram_size,
            renormalize_logits,
            suppress_pad_token,
            prompt_style,
            prompt_format,
            enable_thinking,
            origin_batch_size=resolved_origin_batch_size,
            fallback_depth=fallback_depth + 1,
            fallback_mode="single_retry",
        )[0]
        retry_response["initial_raw_output"] = responses[0]["raw_output"]
        retry_response["initial_raw_output_with_special_tokens"] = responses[0]["raw_output_with_special_tokens"]
        retry_response["initial_effective_batch_size"] = responses[0]["effective_batch_size"]
        retry_response["recovered_after_blank"] = bool(retry_response["raw_output"])
        return [retry_response]

    if len(blank_indices) == len(texts):
        # Cuando todo el lote sale vacío, dividimos recursivamente para reducir la
        # presión de memoria o el efecto de padding antes de culpar al modelo.
        midpoint = max(len(texts) // 2, 1)
        left_responses = generate_responses_batch(
            model,
            tokenizer,
            chat_template_source,
            texts[:midpoint],
            max_new_tokens,
            retry_min_new_tokens,
            retry_temperature,
            retry_top_p,
            top_k,
            repetition_penalty,
            no_repeat_ngram_size,
            renormalize_logits,
            suppress_pad_token,
            fallback_min_new_tokens,
            fallback_temperature,
            fallback_top_p,
            prompt_style,
            prompt_format,
            enable_thinking,
            origin_batch_size=resolved_origin_batch_size,
            fallback_depth=fallback_depth + 1,
        )
        right_responses = generate_responses_batch(
            model,
            tokenizer,
            chat_template_source,
            texts[midpoint:],
            max_new_tokens,
            retry_min_new_tokens,
            retry_temperature,
            retry_top_p,
            top_k,
            repetition_penalty,
            no_repeat_ngram_size,
            renormalize_logits,
            suppress_pad_token,
            fallback_min_new_tokens,
            fallback_temperature,
            fallback_top_p,
            prompt_style,
            prompt_format,
            enable_thinking,
            origin_batch_size=resolved_origin_batch_size,
            fallback_depth=fallback_depth + 1,
        )
        recovered_responses = left_responses + right_responses
        for original_response, recovered_response in zip(responses, recovered_responses):
            recovered_response.setdefault("initial_raw_output", original_response["raw_output"])
            recovered_response.setdefault(
                "initial_raw_output_with_special_tokens",
                original_response["raw_output_with_special_tokens"],
            )
            recovered_response.setdefault("initial_effective_batch_size", original_response["effective_batch_size"])
            recovered_response["recovered_after_blank"] = bool(recovered_response["raw_output"])
        return recovered_responses

    retry_texts = [texts[index] for index in blank_indices]
    retry_responses = generate_responses_batch(
        model,
        tokenizer,
        chat_template_source,
        retry_texts,
        max_new_tokens,
        retry_min_new_tokens,
        retry_temperature,
        retry_top_p,
        top_k,
        repetition_penalty,
        no_repeat_ngram_size,
        renormalize_logits,
        suppress_pad_token,
        fallback_min_new_tokens,
        fallback_temperature,
        fallback_top_p,
        prompt_style,
        prompt_format,
        enable_thinking,
        origin_batch_size=resolved_origin_batch_size,
        fallback_depth=fallback_depth + 1,
    )
    for index, retry_response in zip(blank_indices, retry_responses):
        retry_response.setdefault("initial_raw_output", responses[index]["raw_output"])
        retry_response.setdefault(
            "initial_raw_output_with_special_tokens",
            responses[index]["raw_output_with_special_tokens"],
        )
        retry_response.setdefault("initial_effective_batch_size", responses[index]["effective_batch_size"])
        retry_response["recovered_after_blank"] = bool(retry_response["raw_output"])
        responses[index] = retry_response
    return responses


def write_subtask1_submission(path: Path, prediction_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for prediction_id in prediction_ids:
            writer.writerow([prediction_id])


def write_subtask2_submission(path: Path, prediction_rows: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for row in prediction_rows:
            writer.writerow(row)


def write_verbose_csv_subtask1(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["record_id", "gold_class_id", "gold_class", "pred_class_id", "pred_class", "raw_output"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_verbose_csv_subtask2(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["record_id", *SUBTASK2_COLUMNS, "gold_types", "pred_types", "raw_output"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_generation_debug_trace(
    path: Path,
    task_name: str,
    record_id: str,
    prompt_style: str,
    generation: dict[str, object],
    parsed: dict[str, object],
    gold_payload: dict[str, object],
) -> None:
    append_jsonl(
        path,
        {
            "record_id": record_id,
            "task": task_name,
            "prompt_style": prompt_style,
            "enable_thinking": generation.get("enable_thinking", False),
            "origin_batch_size": generation["origin_batch_size"],
            "effective_batch_size": generation["effective_batch_size"],
            "fallback_depth": generation["fallback_depth"],
            "fallback_mode": generation["fallback_mode"],
            "recovered_after_blank": generation.get("recovered_after_blank", False),
            "input_token_count": generation["input_token_count"],
            "top_k": generation.get("top_k", 0),
            "repetition_penalty": generation.get("repetition_penalty", 1.0),
            "no_repeat_ngram_size": generation.get("no_repeat_ngram_size", 0),
            "renormalize_logits": generation.get("renormalize_logits", False),
            "suppress_pad_token": generation.get("suppress_pad_token", False),
            "prompt_text": generation["prompt_text"],
            "initial_raw_output": generation.get("initial_raw_output", ""),
            "initial_raw_output_with_special_tokens": generation.get(
                "initial_raw_output_with_special_tokens", ""
            ),
            "initial_effective_batch_size": generation.get("initial_effective_batch_size"),
            "raw_output": generation["raw_output"],
            "raw_output_with_special_tokens": generation["raw_output_with_special_tokens"],
            "json_candidate": extract_json_candidate(str(generation["raw_output"])),
            "parsed": parsed,
            **gold_payload,
        },
    )


def main() -> None:
    args = parse_args()
    configure_hf_backend()

    if args.batch_size < 1:
        raise ValueError("--batch-size debe ser >= 1")
    if args.report_every < 1:
        raise ValueError("--report-every debe ser >= 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Cargando modelo ===")
    model, tokenizer, chat_template_source = load_generation_stack(args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    resolved_prompt_format = resolve_prompt_format(chat_template_source, args.prompt_format)

    print(f"backend: {args.inference_backend}")
    print(f"prompt_format: {resolved_prompt_format}")
    if args.inference_backend == "transformers":
        print(f"transformers_dtype: {args.transformers_dtype}")
        print(f"attn_implementation: {args.attn_implementation}")
    print(f"top_k: {args.top_k}")
    print(f"chat_template_disponible: {chat_template_source is not None}")

    selected_record_ids = load_record_ids(args.record_ids_path)
    if selected_record_ids is not None:
        print(f"filtro de record_id activo: {len(selected_record_ids)} ids")

    subtask1_records = slice_records(
        filter_records_by_id(load_subtask1(args.subtask1_split, args.data_dir), selected_record_ids),
        args.limit_subtask1,
    )
    subtask2_records = slice_records(
        filter_records_by_id(load_subtask2(args.subtask2_split, args.data_dir), selected_record_ids),
        args.limit_subtask2,
    )

    subtask1_jsonl = output_dir / "subtask1_predictions.jsonl"
    subtask2_jsonl = output_dir / "subtask2_predictions.jsonl"
    subtask1_debug_jsonl = output_dir / "subtask1_generation_debug.jsonl"
    subtask2_debug_jsonl = output_dir / "subtask2_generation_debug.jsonl"
    subtask1_processed = load_processed_ids(subtask1_jsonl) if args.resume else set()
    subtask2_processed = load_processed_ids(subtask2_jsonl) if args.resume else set()
    pending_subtask1 = [record for record in subtask1_records if record.record_id not in subtask1_processed]
    pending_subtask2 = [record for record in subtask2_records if record.record_id not in subtask2_processed]
    subtask1_debug_written = 0
    subtask2_debug_written = 0

    print("=== Subtask 1 ===")
    subtask1_blank_batch_streak = 0
    subtask1_completed = len(subtask1_processed)
    for batch in batched_records(pending_subtask1, args.batch_size):
        generation_results = generate_responses_batch(
            model,
            tokenizer,
            chat_template_source,
            [record.text for record in batch],
            args.max_new_tokens,
            args.min_new_tokens,
            args.temperature,
            args.top_p,
            args.top_k,
            args.repetition_penalty,
            args.no_repeat_ngram_size,
            args.renormalize_logits,
            args.suppress_pad_token,
            args.fallback_min_new_tokens,
            args.fallback_temperature,
            args.fallback_top_p,
            args.prompt_style,
            resolved_prompt_format,
            args.enable_thinking,
        )
        raw_outputs = [str(result["raw_output"]) for result in generation_results]
        if all(not raw_output for raw_output in raw_outputs):
            subtask1_blank_batch_streak += 1
        else:
            subtask1_blank_batch_streak = 0
        if args.max_consecutive_blank_batches and subtask1_blank_batch_streak >= args.max_consecutive_blank_batches:
            raise RuntimeError(
                "Se abortó la evaluación: demasiados lotes consecutivos con salidas vacías en subtask1."
            )

        for record, generation, raw_output in zip(batch, generation_results, raw_outputs):
            parsed = parse_joint_prediction(raw_output)
            append_jsonl(
                subtask1_jsonl,
                {
                    "record_id": record.record_id,
                    "gold_class_id": record.severity_id,
                    "gold_class": SUBTASK1_ID_TO_NAME[record.severity_id],
                    "pred_class": parsed["severity"] or args.severity_fallback,
                    "pred_types": parsed["violence_types"],
                    "raw_output": raw_output,
                },
            )
            if args.debug_trace_limit > 0 and subtask1_debug_written < args.debug_trace_limit:
                append_generation_debug_trace(
                    subtask1_debug_jsonl,
                    "subtask1",
                    record.record_id,
                    args.prompt_style,
                    generation,
                    parsed,
                    {
                        "gold_class_id": record.severity_id,
                        "gold_class": SUBTASK1_ID_TO_NAME[record.severity_id],
                    },
                )
                subtask1_debug_written += 1

        subtask1_completed += len(batch)
        if should_report_progress(subtask1_completed, len(subtask1_records), args.report_every, len(batch)):
            print(f"subtask1 procesados: {subtask1_completed}/{len(subtask1_records)}")

    print("=== Subtask 2 ===")
    subtask2_blank_batch_streak = 0
    subtask2_completed = len(subtask2_processed)
    for batch in batched_records(pending_subtask2, args.batch_size):
        generation_results = generate_responses_batch(
            model,
            tokenizer,
            chat_template_source,
            [record.text for record in batch],
            args.max_new_tokens,
            args.min_new_tokens,
            args.temperature,
            args.top_p,
            args.top_k,
            args.repetition_penalty,
            args.no_repeat_ngram_size,
            args.renormalize_logits,
            args.suppress_pad_token,
            args.fallback_min_new_tokens,
            args.fallback_temperature,
            args.fallback_top_p,
            args.prompt_style,
            resolved_prompt_format,
            args.enable_thinking,
        )
        raw_outputs = [str(result["raw_output"]) for result in generation_results]
        if all(not raw_output for raw_output in raw_outputs):
            subtask2_blank_batch_streak += 1
        else:
            subtask2_blank_batch_streak = 0
        if args.max_consecutive_blank_batches and subtask2_blank_batch_streak >= args.max_consecutive_blank_batches:
            raise RuntimeError(
                "Se abortó la evaluación: demasiados lotes consecutivos con salidas vacías en subtask2."
            )

        for record, generation, raw_output in zip(batch, generation_results, raw_outputs):
            parsed = parse_joint_prediction(raw_output)
            append_jsonl(
                subtask2_jsonl,
                {
                    "record_id": record.record_id,
                    "gold_types": label_names_from_vector(record.label_vector),
                    "gold_vector": [record.label_vector[column] for column in SUBTASK2_COLUMNS],
                    "pred_class": parsed["severity"],
                    "pred_types": parsed["violence_types"],
                    "raw_output": raw_output,
                },
            )
            if args.debug_trace_limit > 0 and subtask2_debug_written < args.debug_trace_limit:
                append_generation_debug_trace(
                    subtask2_debug_jsonl,
                    "subtask2",
                    record.record_id,
                    args.prompt_style,
                    generation,
                    parsed,
                    {
                        "gold_types": label_names_from_vector(record.label_vector),
                        "gold_vector": [record.label_vector[column] for column in SUBTASK2_COLUMNS],
                    },
                )
                subtask2_debug_written += 1

        subtask2_completed += len(batch)
        if should_report_progress(subtask2_completed, len(subtask2_records), args.report_every, len(batch)):
            print(f"subtask2 procesados: {subtask2_completed}/{len(subtask2_records)}")

    subtask1_rows = [json.loads(line) for line in subtask1_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    subtask2_rows = [json.loads(line) for line in subtask2_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    subtask1_rows_by_id = {row["record_id"]: row for row in subtask1_rows}
    subtask2_rows_by_id = {row["record_id"]: row for row in subtask2_rows}

    ordered_subtask1 = [subtask1_rows_by_id[record.record_id] for record in subtask1_records if record.record_id in subtask1_rows_by_id]
    ordered_subtask2 = [subtask2_rows_by_id[record.record_id] for record in subtask2_records if record.record_id in subtask2_rows_by_id]

    subtask1_gold = [row["gold_class"] for row in ordered_subtask1]
    subtask1_pred = [row["pred_class"] for row in ordered_subtask1]
    subtask1_metrics = compute_multiclass_metrics(subtask1_gold, subtask1_pred, SUBTASK1_LABELS)

    subtask2_gold = [row["gold_vector"] for row in ordered_subtask2]
    subtask2_pred_vectors = []
    verbose_subtask2_rows = []
    for row in ordered_subtask2:
        pred_types = row["pred_types"]
        pred_vector = [1 if label in pred_types else 0 for label in SUBTASK2_LABELS]
        subtask2_pred_vectors.append(pred_vector)
        verbose_row = {
            "record_id": row["record_id"],
            "gold_types": ",".join(row["gold_types"]),
            "pred_types": ",".join(pred_types),
            "raw_output": row["raw_output"],
        }
        for column, value in zip(SUBTASK2_COLUMNS, pred_vector):
            verbose_row[column] = value
        verbose_subtask2_rows.append(verbose_row)

    subtask2_metrics = compute_multilabel_metrics(subtask2_gold, subtask2_pred_vectors, SUBTASK2_LABELS)

    verbose_subtask1_rows = []
    subtask1_submission = []
    for row in ordered_subtask1:
        pred_class_id = SUBTASK1_NAME_TO_ID[row["pred_class"]]
        subtask1_submission.append(pred_class_id)
        verbose_subtask1_rows.append(
            {
                "record_id": row["record_id"],
                "gold_class_id": SUBTASK1_NAME_TO_ID[row["gold_class"]],
                "gold_class": row["gold_class"],
                "pred_class_id": pred_class_id,
                "pred_class": row["pred_class"],
                "raw_output": row["raw_output"],
            }
        )

    submission_dir = output_dir / "submission"
    write_subtask1_submission(submission_dir / "subtask1.csv", subtask1_submission)
    write_subtask2_submission(submission_dir / "subtask2.csv", subtask2_pred_vectors)
    write_verbose_csv_subtask1(output_dir / "subtask1_verbose.csv", verbose_subtask1_rows)
    write_verbose_csv_subtask2(output_dir / "subtask2_verbose.csv", verbose_subtask2_rows)

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "inference_backend": args.inference_backend,
        "record_ids_path": args.record_ids_path,
        "prompt_style": args.prompt_style,
        "enable_thinking": args.enable_thinking,
        "prompt_format": args.prompt_format,
        "resolved_prompt_format": resolved_prompt_format,
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "renormalize_logits": args.renormalize_logits,
        "suppress_pad_token": args.suppress_pad_token,
        "fallback_min_new_tokens": args.fallback_min_new_tokens,
        "fallback_temperature": args.fallback_temperature,
        "fallback_top_p": args.fallback_top_p,
        "debug_trace_limit": args.debug_trace_limit,
        "transformers_dtype": args.transformers_dtype,
        "attn_implementation": args.attn_implementation,
        "subtask1_split": args.subtask1_split,
        "subtask2_split": args.subtask2_split,
        "subtask1_records": len(ordered_subtask1),
        "subtask2_records": len(ordered_subtask2),
        "subtask1_metrics": subtask1_metrics,
        "subtask2_metrics": subtask2_metrics,
    }
    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
