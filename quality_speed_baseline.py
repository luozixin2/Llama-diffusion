"""
Quick quality/speed baseline for micro-block decoding.

Default:
- block_length in {4}, micro_block_size in {2, 4}, gen_length=512
- Prompt list includes the music-robot paragraph used previously.

It prints a human-readable summary and saves JSON + text reports under
`profile_runs/` with a timestamp. Quality is for manual reading; speed is
measured as wall time and tokens/sec (by default: **generated tokens/sec**,
excluding the prompt tokens when the backend returns prompt+gen tokens).
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import llama_diffusion
from transformers import AutoTokenizer


def build_prompts(tokenizer) -> List[Dict[str, Any]]:
    """Return a list of prompts (text + encoded ids)."""
    prompts = [
        {
            "name": "robot-music",
            "messages": [
                {"role": "user", "content": "Write a short paragraph about a robot finding music."}
            ],
        },
        {
            "name": "qa-facts",
            "messages": [
                {"role": "user", "content": "Explain why the sky is blue in two sentences."}
            ],
        },
    ]
    for p in prompts:
        text = tokenizer.apply_chat_template(p["messages"], add_generation_prompt=True, tokenize=False)
        p["text"] = text
        p["prompt_ids"] = tokenizer.encode(text, add_special_tokens=False)
    return prompts


def _extract_assistant_text(decoded: str) -> str:
    # Keep it simple and robust to the chat template differences.
    if "\nassistant" in decoded:
        return decoded.split("\nassistant", 1)[1].strip()
    return decoded.strip()

def _filter_token_id(token_ids: List[int], token_id: int | None) -> List[int]:
    if token_id is None:
        return token_ids
    return [t for t in token_ids if t != token_id]

def _clean_generated_text(text: str) -> str:
    # Remove common control tokens and diffusion masks for readability.
    for t in (
        "<|MASK|>",
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "</s>",
        "<s>",
    ):
        text = text.replace(t, "")
    return text.strip()


def _repetition_metrics(text: str) -> Dict[str, float]:
    # Heuristic metrics: adjacent duplicate word rate + max run length.
    words = [w for w in text.split() if w]
    if len(words) < 2:
        return {"dup_word_rate": 0.0, "max_dup_run": 0.0}

    dup = 0
    max_run = 1
    run = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            dup += 1
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1

    return {
        "dup_word_rate": dup / (len(words) - 1),
        "max_dup_run": float(max_run),
    }


def run_case(
    model,
    tokenizer,
    prompt_entry: Dict[str, Any],
    block_length: int,
    micro_block_size: int,
    gen_length: int,
    use_gpu_sampler: bool,
    denoising_steps: int,
    temperature: float,
    top_k: int,
    top_p: float,
    confidence_threshold: float,
    repetition_penalty: float,
    remasking_strategy: str,
) -> Dict[str, Any]:
    # NOTE:
    # Diffusion generation returns prompt+gen tokens. For correctness, we should always slice by token length
    # rather than attempting to align via decoded text (chat templates + skip_special_tokens can break that).
    mask_id = tokenizer.convert_tokens_to_ids(getattr(tokenizer, "mask_token", None))
    if mask_id is None or mask_id < 0:
        raise RuntimeError(
            "Tokenizer does not define a valid mask_token. "
            "Please pass a tokenizer that contains the diffusion mask token (e.g. <|MASK|>)."
        )
    eos_id = tokenizer.eos_token_id

    start = time.perf_counter()
    # Note: the profiled backend currently does not accept repetition_penalty.
    # For quality comparison vs test_profiling.py (which uses profiled backend), keep repetition_penalty=1.0 there.
    if hasattr(model, "generate_with_profiling"):
        out_tokens, _profile = model.generate_with_profiling(
            prompt=prompt_entry["prompt_ids"],
            mask_token_id=mask_id,
            gen_length=gen_length,
            block_length=block_length,
            denoising_steps=denoising_steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            remasking_strategy=remasking_strategy,
            confidence_threshold=confidence_threshold,
            stop_token_ids=[eos_id] if os.environ.get("DIFFUSION_USE_STOP_EOS", "0") in ("1", "true", "True") else [],
            use_gpu_sampler=use_gpu_sampler,
            micro_block_size=micro_block_size,
        )
    else:
        out_tokens = model.generate(
            prompt=prompt_entry["prompt_ids"],
            mask_token_id=mask_id,
            gen_length=gen_length,
            block_length=block_length,
            micro_block_size=micro_block_size,
            denoising_steps=denoising_steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            remasking_strategy=remasking_strategy,
            confidence_threshold=confidence_threshold,
            repetition_penalty=repetition_penalty,
            stop_token_ids=[eos_id] if os.environ.get("DIFFUSION_USE_STOP_EOS", "0") in ("1", "true", "True") else [],
            use_gpu_sampler=use_gpu_sampler,
        )
    elapsed = time.perf_counter() - start

    prompt_ids = prompt_entry["prompt_ids"]
    prompt_len = len(prompt_ids)
    total_tokens = len(out_tokens)

    # Token-accurate generated part (do NOT truncate at EOS: diffusion may place EOS-like tokens transiently).
    gen_tokens_list = out_tokens[prompt_len:] if total_tokens >= prompt_len else out_tokens
    generated_tokens = len(gen_tokens_list)

    # Decode WITHOUT dropping special tokens; we can post-filter known control tokens.
    decoded_full = tokenizer.decode(out_tokens, skip_special_tokens=False)
    # For readability, filter out EOS token id (if any) but keep the rest.
    decoded_gen = tokenizer.decode(_filter_token_id(gen_tokens_list, eos_id), skip_special_tokens=False)
    decoded_gen_clean = _clean_generated_text(decoded_gen)
    assistant_text = decoded_gen_clean
    rep = _repetition_metrics(assistant_text)

    gen_tokens_per_sec = generated_tokens / elapsed if elapsed > 0 else 0.0
    return {
        "prompt_name": prompt_entry["name"],
        "block_length": block_length,
        "micro_block_size": micro_block_size,
        "denoising_steps": denoising_steps,
        "gen_length": gen_length,
        "use_gpu_sampler": use_gpu_sampler,
        "elapsed_sec": elapsed,
        "prompt_tokens": prompt_len,
        "total_tokens": total_tokens,
        "generated_tokens": generated_tokens,
        "tokens": total_tokens,  # backward-compat alias
        "tokens_per_sec": gen_tokens_per_sec,  # default: generated tokens/sec
        "gen_tokens_per_sec": gen_tokens_per_sec,
        "dup_word_rate": rep["dup_word_rate"],
        "max_dup_run": rep["max_dup_run"],
        # Print-friendly view first; keep debug fields for inspection.
        "output_text": decoded_gen_clean,
        "output_text_debug_full": decoded_full,
        "output_text_debug_gen": decoded_gen,
    }


def main():
    parser = argparse.ArgumentParser(description="Quality/Speed baseline for micro-block decoding")
    parser.add_argument("--model-path", default="/home/lzx/SDAR/training/model/SDAR-1.7B-Chat/SDAR-1.7B-Chat-F16.gguf")
    parser.add_argument("--tokenizer-path", default="/home/lzx/SDAR/training/model/SDAR-1.7B-Chat")
    parser.add_argument("--gen-length", type=int, default=512)
    parser.add_argument("--denoising-steps", type=int, default=0, help="If 0, auto-set to min(block_length, 8)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument(
        "--remasking-strategy",
        type=str,
        default="low_confidence_dynamic",
        choices=["sequential", "low_confidence_static", "low_confidence_dynamic", "entropy_bounded"],
    )
    parser.add_argument("--use-gpu-sampler", action="store_true", default=False)
    parser.add_argument(
        "--profiled-backend",
        action="store_true",
        default=False,
        help="Use the profiled backend (llama_diffusion_profiled) for quality validation. "
             "This matches test_profiling.py, but currently does not support repetition_penalty.",
    )
    parser.add_argument("--seed", type=int, default=-1, help="If set >=0, export DIFFUSION_SEED for reproducibility")
    parser.add_argument("--partial-kv", action="store_true", default=False, help="Enable partial-KV reuse (quality may degrade)")
    parser.add_argument("--force-full-decode", action="store_true", default=False, help="Force full block decode (disable partial-KV)")
    parser.add_argument("--block-lengths", type=int, nargs="+", default=[4])
    parser.add_argument("--micro-block-sizes", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--n-gpu-layers", type=int, default=35)
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--use-stop-eos", action="store_true", default=False,
                        help="Enable EOS early stop (not recommended for diffusion; can stop too early)")
    parser.add_argument("--freeze-done-micro", action="store_true", default=False,
                        help="Enable freezing finished micro-blocks (perf feature; can hurt quality for small micro sizes)")
    parser.add_argument("--done-micro-no-logits", action="store_true", default=False,
                        help="When freezing is enabled, do not request logits for frozen micro-blocks (max perf; may hurt quality)")
    args = parser.parse_args()

    # Set optimization environment variables
    # GPU sampler optimizations: device logits for faster GPU sampling
    if args.use_gpu_sampler:
        os.environ.setdefault("LLAMA_ENABLE_DEVICE_LOGITS", "1")
        # Default to sync mode for quality stability; profiling scripts can override to "1".
        os.environ.setdefault("LLAMA_DEVICE_LOGITS_ASYNC", "0")
    
    # Skip synchronization after get_output_ids to reduce host overhead
    # NOTE: Setting this to "1" can improve perf but may degrade quality when device logits are async.
    os.environ.setdefault("DIFFUSION_SKIP_SYNC_AFTER_OUTPUT_IDS", "0")
    
    if args.seed >= 0:
        os.environ["DIFFUSION_SEED"] = str(args.seed)
    os.environ["DIFFUSION_USE_STOP_EOS"] = "1" if args.use_stop_eos else "0"

    # Quality-first defaults: keep micro-freeze OFF unless explicitly requested.
    os.environ["DIFFUSION_FREEZE_DONE_MICRO"] = "1" if args.freeze_done_micro else "0"
    os.environ["DIFFUSION_DONE_MICRO_NO_LOGITS"] = "1" if args.done_micro_no_logits else "0"

    # Partial-KV reuse toggle:
    # - Default OFF for quality stability.
    # - When GPU sampler is enabled, partial-KV also requires DIFFUSION_PARTIAL_KV_REUSE_GPU=1.
    if args.force_full_decode:
        os.environ["DIFFUSION_FORCE_FULL_BLOCK_DECODE"] = "1"
        os.environ["DIFFUSION_PARTIAL_KV_REUSE"] = "0"
        os.environ["DIFFUSION_PARTIAL_KV_REUSE_GPU"] = "0"
    elif args.partial_kv:
        os.environ["DIFFUSION_PARTIAL_KV_REUSE"] = "1"
        os.environ["DIFFUSION_PARTIAL_KV_REUSE_GPU"] = "1" if args.use_gpu_sampler else "0"
    else:
        os.environ["DIFFUSION_PARTIAL_KV_REUSE"] = "0"
        os.environ["DIFFUSION_PARTIAL_KV_REUSE_GPU"] = "0"

    # Load tokenizer/model once
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    print("Loading model...")
    if args.profiled_backend:
        from llama_diffusion.llama_diffusion_profiled import LlamaDiffusionProfiled
        model = LlamaDiffusionProfiled(
            model_path=args.model_path,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
        )
        if args.repetition_penalty != 1.0:
            print("[warn] --profiled-backend currently ignores repetition_penalty; please use --repetition-penalty 1.0 for apples-to-apples.")
    else:
        model = llama_diffusion.LlamaDiffusion(
            model_path=args.model_path,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
        )

    prompts = build_prompts(tokenizer)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("profile_runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"quality_speed_baseline_{timestamp}.json"
    txt_path = out_dir / f"quality_speed_baseline_{timestamp}.txt"

    results = []
    lines = []
    lines.append(f"LLAMA_ENABLE_DEVICE_LOGITS={os.environ.get('LLAMA_ENABLE_DEVICE_LOGITS')}")
    lines.append(f"LLAMA_DEVICE_LOGITS_ASYNC={os.environ.get('LLAMA_DEVICE_LOGITS_ASYNC')}")
    lines.append(f"DIFFUSION_SKIP_SYNC_AFTER_OUTPUT_IDS={os.environ.get('DIFFUSION_SKIP_SYNC_AFTER_OUTPUT_IDS')}")
    lines.append(f"DIFFUSION_PARTIAL_KV_REUSE={os.environ.get('DIFFUSION_PARTIAL_KV_REUSE')}")
    lines.append(f"DIFFUSION_PARTIAL_KV_REUSE_GPU={os.environ.get('DIFFUSION_PARTIAL_KV_REUSE_GPU')}")
    lines.append(f"DIFFUSION_FORCE_FULL_BLOCK_DECODE={os.environ.get('DIFFUSION_FORCE_FULL_BLOCK_DECODE')}")
    lines.append(f"DIFFUSION_SEED={os.environ.get('DIFFUSION_SEED')}")
    lines.append(f"use_gpu_sampler={args.use_gpu_sampler}")
    lines.append(f"block_lengths={args.block_lengths}, micro_block_sizes={args.micro_block_sizes}, gen_length={args.gen_length}")
    lines.append("")

    for b in args.block_lengths:
        # Auto-set denoising_steps to a quality-friendly default if not explicitly provided
        denoising_steps = args.denoising_steps if args.denoising_steps > 0 else min(b, 8)
        for m in args.micro_block_sizes:
            if b % m != 0:
                print(f"Skip: block_length {b} not divisible by micro_block_size {m}")
                continue
            for p in prompts:
                print(f"\n=== Running prompt={p['name']} block={b} micro={m} steps={denoising_steps} ===")
                res = run_case(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_entry=p,
                    block_length=b,
                    micro_block_size=m,
                    gen_length=args.gen_length,
                    use_gpu_sampler=args.use_gpu_sampler,
                    denoising_steps=denoising_steps,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    confidence_threshold=args.confidence_threshold,
                    repetition_penalty=args.repetition_penalty,
                    remasking_strategy=args.remasking_strategy,
                )
                results.append(res)
                lines.append(
                    f"[{p['name']}] block={b} micro={m} steps={denoising_steps} "
                    f"elapsed={res['elapsed_sec']:.2f}s gen_tps={res['gen_tokens_per_sec']:.2f} "
                    f"gen={res['generated_tokens']} total={res['total_tokens']} prompt={res['prompt_tokens']} "
                    f"dup_rate={res['dup_word_rate']:.3f} max_run={int(res['max_dup_run'])}"
                )
                lines.append(res["output_text"])
                lines.append("")

    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    txt_path.write_text("\n".join(lines))
    print(f"\nSaved results to:\n  {json_path}\n  {txt_path}")


if __name__ == "__main__":
    main()

