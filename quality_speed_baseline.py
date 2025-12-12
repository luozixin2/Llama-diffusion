"""
Quick quality/speed baseline for micro-block decoding.

Default:
- block_length in {4, 8}, micro_block_size in {2, 4}, gen_length=512
- Prompt list includes the music-robot paragraph used previously.

It prints a human-readable summary and saves JSON + text reports under
`profile_runs/` with a timestamp. Quality is for manual reading; speed is
measured as wall time and tokens/sec.
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
    tokens_per_sec = len(out_tokens) / elapsed if elapsed > 0 else 0.0
    decoded = tokenizer.decode(out_tokens, skip_special_tokens=True)
    return {
        "prompt_name": prompt_entry["name"],
        "block_length": block_length,
        "micro_block_size": micro_block_size,
        "gen_length": gen_length,
        "use_gpu_sampler": use_gpu_sampler,
        "elapsed_sec": elapsed,
        "tokens": len(out_tokens),
        "tokens_per_sec": tokens_per_sec,
        "output_text": decoded,
    }


def main():
    parser = argparse.ArgumentParser(description="Quality/Speed baseline for micro-block decoding")
    parser.add_argument("--model-path", default="/home/lzx/SDAR/training/model/SDAR-1.7B-Chat/SDAR-1.7B-Chat-F16.gguf")
    parser.add_argument("--tokenizer-path", default="/home/lzx/SDAR/training/model/SDAR-1.7B-Chat")
    parser.add_argument("--gen-length", type=int, default=512)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--use-gpu-sampler", action="store_true", default=False)
    parser.add_argument("--block-lengths", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--micro-block-sizes", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--n-gpu-layers", type=int, default=35)
    parser.add_argument("--n-ctx", type=int, default=8192)
    args = parser.parse_args()

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
    lines.append(f"use_gpu_sampler={args.use_gpu_sampler}")
    lines.append(f"block_lengths={args.block_lengths}, micro_block_sizes={args.micro_block_sizes}, gen_length={args.gen_length}")
    lines.append("")

    for b in args.block_lengths:
        for m in args.micro_block_sizes:
            if b % m != 0:
                print(f"Skip: block_length {b} not divisible by micro_block_size {m}")
                continue
            for p in prompts:
                print(f"\n=== Running prompt={p['name']} block={b} micro={m} ===")
                res = run_case(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_entry=p,
                    block_length=b,
                    micro_block_size=m,
                    gen_length=args.gen_length,
                    use_gpu_sampler=args.use_gpu_sampler,
                    denoising_steps=args.denoising_steps,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )
                results.append(res)
                lines.append(
                    f"[{p['name']}] block={b} micro={m} "
                    f"elapsed={res['elapsed_sec']:.2f}s tps={res['tokens_per_sec']:.2f} "
                    f"len={res['tokens']}"
                )
                lines.append(res["output_text"])
                lines.append("")

    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    txt_path.write_text("\n".join(lines))
    print(f"\nSaved results to:\n  {json_path}\n  {txt_path}")


if __name__ == "__main__":
    main()

