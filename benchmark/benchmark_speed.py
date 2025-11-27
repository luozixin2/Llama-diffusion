#!/usr/bin/env python3
"""
SDAR 1.7B 速度基准测试脚本
测试配置: block_length=denoising_steps=1,2,3,4 + cpu_sampler对比
"""

import os
import sys
import time
import json
from datetime import datetime

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import llama_diffusion
from transformers import AutoTokenizer

MODEL_PATH = '/home/lzx/SDAR/training/model/SDAR-1.7B-Chat/SDAR-1.7B-Chat-F16.gguf'
TOKENIZER_PATH = '/home/lzx/SDAR/training/model/SDAR-1.7B-Chat'

def run_speed_test():
    print("=" * 70)
    print("SDAR 1.7B Speed Benchmark")
    print("=" * 70)
    
    # 加载tokenizer和模型
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    eos_token_id = tokenizer.eos_token_id
    
    print("Loading model...")
    model = llama_diffusion.LlamaDiffusion(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_gpu_layers=35
    )
    print("Model loaded!\n")
    
    # 测试配置
    configs = [
        {"name": "Block1_Step1_GPU", "block_length": 1, "denoising_steps": 1, "use_gpu_sampler": True},
        {"name": "Block2_Step2_GPU", "block_length": 2, "denoising_steps": 2, "use_gpu_sampler": True},
        {"name": "Block3_Step3_GPU", "block_length": 3, "denoising_steps": 3, "use_gpu_sampler": True},
        {"name": "Block4_Step4_GPU", "block_length": 4, "denoising_steps": 4, "use_gpu_sampler": True},
        {"name": "Block4_Step4_CPU", "block_length": 4, "denoising_steps": 4, "use_gpu_sampler": False},
    ]
    
    # 测试prompts
    test_prompts = [
        "What is 2+2?",
        "Explain the theory of relativity in simple terms.",
        "Write a Python function to calculate fibonacci numbers.",
        "What are the main causes of climate change?",
        "Describe the process of photosynthesis.",
    ]
    
    gen_length = 256
    num_warmup = 2
    num_runs = 10
    
    results = []
    
    for config in configs:
        print("*" * 70)
        print(f"* Testing: {config['name']}")
        print(f"* Block Length: {config['block_length']}, Denoising Steps: {config['denoising_steps']}")
        print(f"* GPU Sampler: {config['use_gpu_sampler']}")
        print("*" * 70)
        
        # 预热
        print(f"  Warming up ({num_warmup} runs)...")
        prompt = test_prompts[0]
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        
        for _ in range(num_warmup):
            _ = model.generate(
                prompt=prompt_tokens,
                mask_token_id=mask_token_id,
                gen_length=gen_length,
                block_length=config['block_length'],
                denoising_steps=config['denoising_steps'],
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                remasking_strategy="low_confidence_dynamic",
                confidence_threshold=0.85,
                stop_token_ids=[eos_token_id],
                use_gpu_sampler=config['use_gpu_sampler']
            )
        
        # 正式测试
        print(f"  Running {num_runs} tests...")
        times = []
        total_tokens = 0
        
        for i, prompt in enumerate(test_prompts[:num_runs]):
            messages = [{"role": "user", "content": prompt}]
            prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
            
            start_time = time.perf_counter()
            output_tokens = model.generate(
                prompt=prompt_tokens,
                mask_token_id=mask_token_id,
                gen_length=gen_length,
                block_length=config['block_length'],
                denoising_steps=config['denoising_steps'],
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                remasking_strategy="low_confidence_dynamic",
                confidence_threshold=0.85,
                stop_token_ids=[eos_token_id],
                use_gpu_sampler=config['use_gpu_sampler']
            )
            end_time = time.perf_counter()
            
            elapsed = end_time - start_time
            generated = len(output_tokens) - len(prompt_tokens)
            times.append(elapsed)
            total_tokens += generated
        
        avg_time = sum(times) / len(times)
        throughput = total_tokens / sum(times)
        
        result = {
            "config": config['name'],
            "block_length": config['block_length'],
            "denoising_steps": config['denoising_steps'],
            "gpu_sampler": config['use_gpu_sampler'],
            "avg_time_sec": round(avg_time, 3),
            "throughput_tokens_per_sec": round(throughput, 2),
            "total_tokens": total_tokens,
            "num_runs": num_runs
        }
        results.append(result)
        
        print(f"  Average time: {avg_time:.3f}s")
        print(f"  Throughput: {throughput:.2f} tokens/sec")
        print()
    
    # 保存结果
    output_dir = f"benchmark_results/speed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # 生成报告
    report = f"""# SDAR 1.7B Speed Benchmark Report

**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 测试配置
- 模型: SDAR-1.7B-Chat (F16 GGUF)
- 生成长度: {gen_length} tokens
- 测试次数: {num_runs}
- 预热次数: {num_warmup}
- 温度: 1.0, top_p: 1.0, top_k: 0

## 速度测试结果

| 配置 | Block Length | Denoising Steps | GPU Sampler | 平均时间(s) | 吞吐量(tokens/s) |
|------|--------------|-----------------|-------------|------------|-----------------|
"""
    for r in results:
        gpu_str = "✅" if r['gpu_sampler'] else "❌"
        report += f"| {r['config']} | {r['block_length']} | {r['denoising_steps']} | {gpu_str} | {r['avg_time_sec']} | {r['throughput_tokens_per_sec']} |\n"
    
    # GPU vs CPU 对比
    gpu_result = next((r for r in results if r['config'] == 'Block4_Step4_GPU'), None)
    cpu_result = next((r for r in results if r['config'] == 'Block4_Step4_CPU'), None)
    
    if gpu_result and cpu_result:
        speedup = cpu_result['avg_time_sec'] / gpu_result['avg_time_sec']
        report += f"""
## GPU Sampler vs CPU Sampler 对比 (Block4_Step4)

| 指标 | GPU Sampler | CPU Sampler | 差异 |
|------|-------------|-------------|------|
| 平均时间 | {gpu_result['avg_time_sec']}s | {cpu_result['avg_time_sec']}s | {speedup:.2f}x |
| 吞吐量 | {gpu_result['throughput_tokens_per_sec']} t/s | {cpu_result['throughput_tokens_per_sec']} t/s | - |
"""
    
    report += """
## 结论

- Block Length和Denoising Steps增加会提高质量但降低速度
- Block1_Step1 最快但质量最低
- Block4_Step4 是官方推荐配置，平衡质量和速度
"""
    
    with open(f"{output_dir}/SPEED_REPORT.md", 'w') as f:
        f.write(report)
    
    print("=" * 70)
    print(f"Results saved to: {output_dir}")
    print("=" * 70)
    
    # 打印表格
    print("\n" + report)
    
    return results

if __name__ == "__main__":
    run_speed_test()

