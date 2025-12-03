#!/usr/bin/env python3
"""
高并发条件下的极限生成速度测试
对比CPU和GPU采样器在不同块大小和步数配置下的吞吐量
"""
from __future__ import annotations

import llama_diffusion
from transformers import AutoTokenizer
import time
import threading
import json
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import os
import shutil


class ConcurrentThroughputTester:
    def __init__(self, model_path: str, n_ctx: int = 8192, n_gpu_layers: int = 35):
        """初始化测试器"""
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.models = []  # 存储多个模型实例用于并发
        
    def create_model(self):
        """创建一个新的模型实例"""
        return llama_diffusion.LlamaDiffusion(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers
        )
    
    def warmup(self, prompt: List[int], mask_token_id: int, num_warmup: int = 3):
        """预热GPU和模型"""
        print(f"\n{'='*80}")
        print(f"WARMUP - Running {num_warmup} iterations")
        print(f"{'='*80}")
        
        model = self.create_model()
        
        for i in range(num_warmup):
            print(f"Warmup iteration {i+1}/{num_warmup}...", end=' ', flush=True)
            start = time.time()
            try:
                tokens = model.generate(
                    prompt=prompt,
                    mask_token_id=mask_token_id,
                    gen_length=64,
                    block_length=4,
                    denoising_steps=4,
                    remasking_strategy='low_confidence_dynamic',
                    use_gpu_sampler=False
                )
                elapsed = (time.time() - start) * 1000
                print(f"completed in {elapsed:.2f} ms, generated {len(tokens)} tokens")
            except Exception as e:
                print(f"failed: {e}")
        
        print(f"{'='*80}")
        print("Warmup completed!")
        print(f"{'='*80}\n")
        time.sleep(1)
    
    def single_generation_task(
        self,
        model: llama_diffusion.LlamaDiffusion,
        prompt: List[int],
        mask_token_id: int,
        gen_length: int,
        block_length: int,
        denoising_steps: int,
        use_gpu_sampler: bool,
        task_id: int
    ) -> Dict:
        """单个生成任务"""
        start_time = time.time()
        try:
            tokens = model.generate(
                prompt=prompt,
                mask_token_id=mask_token_id,
                gen_length=gen_length,
                block_length=block_length,
                denoising_steps=denoising_steps,
                remasking_strategy='low_confidence_dynamic',
                use_gpu_sampler=use_gpu_sampler
            )
            end_time = time.time()
            
            elapsed_ms = (end_time - start_time) * 1000
            tokens_per_sec = len(tokens) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
            
            return {
                'task_id': task_id,
                'success': True,
                'tokens_generated': len(tokens),
                'wall_time_ms': elapsed_ms,
                'throughput_tokens_per_sec': tokens_per_sec,
                'error': None
            }
        except Exception as e:
            end_time = time.time()
            elapsed_ms = (end_time - start_time) * 1000
            return {
                'task_id': task_id,
                'success': False,
                'tokens_generated': 0,
                'wall_time_ms': elapsed_ms,
                'throughput_tokens_per_sec': 0,
                'error': str(e)
            }
    
    def run_concurrent_test(
        self,
        prompt: List[int],
        mask_token_id: int,
        gen_length: int,
        block_length: int,
        denoising_steps: int,
        use_gpu_sampler: bool,
        num_concurrent: int = 4,
        num_rounds: int = 3
    ) -> Dict:
        """运行并发测试"""
        print(f"\n{'='*80}")
        config_name = f"block={block_length}, steps={denoising_steps}, {'GPU' if use_gpu_sampler else 'CPU'}"
        print(f"Running concurrent test: {config_name}")
        print(f"Concurrent tasks: {num_concurrent}, Rounds: {num_rounds}")
        print(f"{'='*80}")
        
        all_results = []
        
        for round_idx in range(num_rounds):
            print(f"\nRound {round_idx + 1}/{num_rounds}...", end=' ', flush=True)
            
            # 创建多个模型实例用于并发
            models = [self.create_model() for _ in range(num_concurrent)]
            
            # 记录开始时间
            round_start_time = time.time()
            
            # 使用线程池执行并发任务
            with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
                futures = []
                for i in range(num_concurrent):
                    future = executor.submit(
                        self.single_generation_task,
                        models[i],
                        prompt,
                        mask_token_id,
                        gen_length,
                        block_length,
                        denoising_steps,
                        use_gpu_sampler,
                        i
                    )
                    futures.append(future)
                
                # 收集结果
                round_results = []
                for future in as_completed(futures):
                    result = future.result()
                    round_results.append(result)
            
            round_end_time = time.time()
            round_wall_time = (round_end_time - round_start_time) * 1000
            
            # 计算统计信息
            successful_results = [r for r in round_results if r['success']]
            total_tokens = sum(r['tokens_generated'] for r in successful_results)
            total_throughput = sum(r['throughput_tokens_per_sec'] for r in successful_results)
            avg_task_time = sum(r['wall_time_ms'] for r in successful_results) / len(successful_results) if successful_results else 0
            
            round_summary = {
                'round': round_idx + 1,
                'wall_time_ms': round_wall_time,
                'num_tasks': num_concurrent,
                'num_successful': len(successful_results),
                'total_tokens': total_tokens,
                'aggregate_throughput_tokens_per_sec': total_tokens / (round_wall_time / 1000) if round_wall_time > 0 else 0,
                'avg_task_throughput_tokens_per_sec': total_throughput / len(successful_results) if successful_results else 0,
                'avg_task_time_ms': avg_task_time,
                'individual_results': round_results
            }
            
            all_results.append(round_summary)
            
            print(f"completed in {round_wall_time:.2f} ms, "
                  f"{len(successful_results)}/{num_concurrent} successful, "
                  f"aggregate throughput: {round_summary['aggregate_throughput_tokens_per_sec']:.2f} tokens/sec")
        
        # 计算平均统计信息
        avg_wall_time = sum(r['wall_time_ms'] for r in all_results) / len(all_results)
        avg_aggregate_throughput = sum(r['aggregate_throughput_tokens_per_sec'] for r in all_results) / len(all_results)
        avg_task_throughput = sum(r['avg_task_throughput_tokens_per_sec'] for r in all_results) / len(all_results)
        total_tokens_all_rounds = sum(r['total_tokens'] for r in all_results)
        total_wall_time_all_rounds = sum(r['wall_time_ms'] for r in all_results)
        overall_throughput = total_tokens_all_rounds / (total_wall_time_all_rounds / 1000) if total_wall_time_all_rounds > 0 else 0
        
        return {
            'config': {
                'block_length': block_length,
                'denoising_steps': denoising_steps,
                'use_gpu_sampler': use_gpu_sampler,
                'gen_length': gen_length,
                'num_concurrent': num_concurrent,
                'num_rounds': num_rounds
            },
            'summary': {
                'avg_wall_time_ms': avg_wall_time,
                'avg_aggregate_throughput_tokens_per_sec': avg_aggregate_throughput,
                'avg_task_throughput_tokens_per_sec': avg_task_throughput,
                'overall_throughput_tokens_per_sec': overall_throughput,
                'total_tokens': total_tokens_all_rounds,
                'total_wall_time_ms': total_wall_time_all_rounds
            },
            'rounds': all_results
        }
    
    def run_all_configs(
        self,
        prompt: List[int],
        mask_token_id: int,
        gen_length: int = 128,
        num_concurrent: int = 4,
        num_rounds: int = 3,
        warmup_before_test: bool = True
    ) -> List[Dict]:
        """运行所有配置的测试"""
        
        # 预热
        if warmup_before_test:
            self.warmup(prompt, mask_token_id)
        
        # 定义所有配置：块=步数=1, 2, 4, 8，CPU和GPU各4种
        configs = []
        for block_steps in [1, 2, 4, 8]:
            # CPU配置
            configs.append({
                'block_length': block_steps,
                'denoising_steps': block_steps,
                'use_gpu_sampler': False,
                'name': f'CPU (block={block_steps}, steps={block_steps})'
            })
            # GPU配置
            configs.append({
                'block_length': block_steps,
                'denoising_steps': block_steps,
                'use_gpu_sampler': True,
                'name': f'GPU (block={block_steps}, steps={block_steps})'
            })
        
        results = []
        
        for i, config in enumerate(configs):
            result = self.run_concurrent_test(
                prompt=prompt,
                mask_token_id=mask_token_id,
                gen_length=gen_length,
                block_length=config['block_length'],
                denoising_steps=config['denoising_steps'],
                use_gpu_sampler=config['use_gpu_sampler'],
                num_concurrent=num_concurrent,
                num_rounds=num_rounds
            )
            result['config']['name'] = config['name']
            results.append(result)
            
            # 配置间短暂等待
            if i < len(configs) - 1:
                time.sleep(1)
        
        return results
    
    def print_summary(self, results: List[Dict]):
        """打印测试结果摘要"""
        print(f"\n{'='*80}")
        print("CONCURRENT THROUGHPUT TEST SUMMARY")
        print(f"{'='*80}\n")
        
        print(f"{'Config':<35} {'Overall Throughput':<20} {'Avg Task Throughput':<20} {'Avg Wall Time':<15}")
        print(f"{'':<35} {'(tokens/sec)':<20} {'(tokens/sec)':<20} {'(ms)':<15}")
        print("-" * 90)
        
        for result in results:
            config_name = result['config']['name']
            overall_throughput = result['summary']['overall_throughput_tokens_per_sec']
            avg_task_throughput = result['summary']['avg_task_throughput_tokens_per_sec']
            avg_wall_time = result['summary']['avg_wall_time_ms']
            
            print(f"{config_name:<35} "
                  f"{overall_throughput:<20.2f} "
                  f"{avg_task_throughput:<20.2f} "
                  f"{avg_wall_time:<15.2f}")
        
        # 找出最快的配置
        fastest = max(results, key=lambda r: r['summary']['overall_throughput_tokens_per_sec'])
        print(f"\n{'='*80}")
        print(f"Fastest Configuration: {fastest['config']['name']}")
        print(f"Overall Throughput: {fastest['summary']['overall_throughput_tokens_per_sec']:.2f} tokens/sec")
        print(f"{'='*80}")
    
    def export_results(self, results: List[Dict], output_file: str = 'concurrent_throughput_results.json', archive: bool = True):
        """导出结果到JSON"""
        export_data = {
            'test_timestamp': datetime.now().isoformat(),
            'results': []
        }
        
        for result in results:
            export_data['results'].append({
                'config': result['config'],
                'summary': result['summary'],
                'rounds': result['rounds']
            })
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\nResults exported to {output_file}")
        
        if archive:
            archive_dir = 'concurrent_throughput_runs'
            if not os.path.exists(archive_dir):
                os.makedirs(archive_dir)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_subdir = os.path.join(archive_dir, timestamp)
            os.makedirs(archive_subdir, exist_ok=True)
            
            archive_json = os.path.join(archive_subdir, output_file)
            shutil.copy2(output_file, archive_json)
            
            print(f"Results archived to {archive_subdir}")


def main():
    """主测试函数"""
    # 配置
    MODEL_PATH = "/home/lzx/SDAR/training/model/SDAR-1.7B-Chat/SDAR-1.7B-Chat-F16.gguf"
    TOKENIZER_PATH = "/home/lzx/SDAR/training/model/SDAR-1.7B-Chat"
    
    # 加载tokenizer
    print("=" * 80)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    MASK_TOKEN_ID = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    print(f"Mask token ID: {MASK_TOKEN_ID}")
    
    # 准备prompt
    messages = [{"role": "user", "content": "写一个关于一个机器人第一次发现音乐的短篇故事。"}]
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    print(f"Prompt length: {len(prompt)} tokens")
    
    # 初始化测试器
    print("\n" + "=" * 80)
    print("Initializing ConcurrentThroughputTester...")
    print("=" * 80)
    tester = ConcurrentThroughputTester(MODEL_PATH, n_ctx=8192, n_gpu_layers=35)
    
    # 运行所有配置的测试
    # 参数说明：
    # - gen_length: 每次生成的token数量
    # - num_concurrent: 并发任务数（同时运行的生成任务数）
    # - num_rounds: 每个配置运行的轮数
    results = tester.run_all_configs(
        prompt=prompt,
        mask_token_id=MASK_TOKEN_ID,
        gen_length=128,
        num_concurrent=4,  # 4个并发任务
        num_rounds=3,      # 每个配置运行3轮
        warmup_before_test=True
    )
    
    # 打印摘要
    tester.print_summary(results)
    
    # 导出结果
    tester.export_results(results)
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()

