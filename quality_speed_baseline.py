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
) -> Dict[str, Any]:
    mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    eos_id = tokenizer.eos_token_id

    start = time.perf_counter()
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
        remasking_strategy="low_confidence_dynamic",
        stop_token_ids=[eos_id],
        use_gpu_sampler=use_gpu_sampler,
    )
    elapsed = time.perf_counter() - start

    # NOTE: C++ DiffusionSampler::generate() returns prompt+gen tokens.
    # For throughput, we report generated tokens/sec (exclude prompt tokens).
    # Calculate actual generated tokens by extracting the generated part from decoded text.
    # This is more accurate than counting tokens in the list, as it handles early stops correctly.
    prompt_ids = prompt_entry["prompt_ids"]
    prompt_len = len(prompt_ids)
    total_tokens = len(out_tokens)
    decoded = tokenizer.decode(out_tokens, skip_special_tokens=True)
    assistant_text = _extract_assistant_text(decoded)
    rep = _repetition_metrics(assistant_text)
    
    # Extract the generated part from decoded text
    # Use same skip_special_tokens setting for both to ensure matching
    prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
    if decoded.startswith(prompt_text):
        # Extract the generated part (after prompt)
        generated_text = decoded[len(prompt_text):].strip()
        # Re-encode to get accurate token count of actual generated content
        generated_tokens = len(tokenizer.encode(generated_text, add_special_tokens=False))
    else:
        # If output doesn't start with prompt, try to find "assistant" marker
        if "assistant" in decoded.lower():
            # Extract text after "assistant" marker
            parts = decoded.split("assistant", 1)
            if len(parts) > 1:
                generated_text = parts[-1].strip()
                generated_tokens = len(tokenizer.encode(generated_text, add_special_tokens=False))
            else:
                # Fallback: count all tokens minus prompt
                generated_tokens = total_tokens - prompt_len if total_tokens >= prompt_len else total_tokens
        else:
            # Fallback: count all tokens minus prompt
            generated_tokens = total_tokens - prompt_len if total_tokens >= prompt_len else total_tokens

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
        "output_text": decoded,
    }


def main():
    parser = argparse.ArgumentParser(description="Quality/Speed baseline for micro-block decoding")
    parser.add_argument("--model-path", default="/home/lzx/SDAR/training/model/SDAR-1.7B-Chat/SDAR-1.7B-Chat-F16.gguf")
    parser.add_argument("--tokenizer-path", default="/home/lzx/SDAR/training/model/SDAR-1.7B-Chat")
    parser.add_argument("--gen-length", type=int, default=512)
    parser.add_argument("--denoising-steps", type=int, default=0, help="If 0, auto-set to block_length (b=s)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--use-gpu-sampler", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=-1, help="If set >=0, export DIFFUSION_SEED for reproducibility")
    parser.add_argument("--partial-kv", action="store_true", default=False, help="Enable partial-KV reuse (quality may degrade)")
    parser.add_argument("--force-full-decode", action="store_true", default=False, help="Force full block decode (disable partial-KV)")
    parser.add_argument("--block-lengths", type=int, nargs="+", default=[4])
    parser.add_argument("--micro-block-sizes", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--n-gpu-layers", type=int, default=35)
    parser.add_argument("--n-ctx", type=int, default=8192)
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
        # Auto-set denoising_steps to block_length if not explicitly provided
        denoising_steps = args.denoising_steps if args.denoising_steps > 0 else b
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

