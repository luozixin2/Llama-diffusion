import llama_diffusion.llama_diffusion_profiled as llama_diffusion_profiled
import json 
import time
from typing import Dict, List
import matplotlib.pyplot as plt
import pandas as pd
import copy

class DiffusionProfiler:
    def __init__(self, model_path: str, n_ctx: int = 8192, n_gpu_layers: int = 0):
        self.model = llama_diffusion_profiled.LlamaDiffusionProfiled(
            model_path, n_ctx, n_gpu_layers
        )
        self.is_warmed_up = False
    
    def warmup(
        self, 
        prompt: List[int],
        mask_token_id: int,
        warmup_iterations: int = 3,
        gen_length: int = 64,
        block_length: int = 8,
        denoising_steps: int = 4
    ):
        """GPU warmup - 运行几次推理以预热GPU和缓存"""
        print(f"\n{'='*80}")
        print(f"GPU WARMUP - Running {warmup_iterations} iterations")
        print(f"{'='*80}")
        
        for i in range(warmup_iterations):
            print(f"Warmup iteration {i+1}/{warmup_iterations}...", end=' ', flush=True)
            start = time.time()
            
            # 运行一次简短的生成（不保存结果）
            try:
                self.model.generate_with_profiling(
                    prompt=prompt,
                    mask_token_id=mask_token_id,
                    gen_length=gen_length,
                    block_length=block_length,
                    denoising_steps=denoising_steps,
                    remasking_strategy='low_confidence_dynamic'
                )
                elapsed = (time.time() - start) * 1000
                print(f"completed in {elapsed:.2f} ms")
            except Exception as e:
                print(f"failed: {e}")
        
        self.is_warmed_up = True
        print(f"{'='*80}")
        print("Warmup completed!")
        print(f"{'='*80}\n")
        
        # 短暂等待让GPU稳定
        time.sleep(1)
    
    def run_profiling_test(
        self,
        prompt: List[int],
        mask_token_id: int,
        gen_length: int = 128,
        block_length: int = 8,
        denoising_steps: int = 8,
        ensure_warmup: bool = True,
        **kwargs
    ):
        """运行单次性能测试"""
        
        # 检查是否需要warmup
        if ensure_warmup and not self.is_warmed_up:
            print("\n⚠️  Warning: GPU not warmed up. Running warmup first...")
            self.warmup(prompt, mask_token_id)
        
        # 复制 kwargs 并移除不传给 C++ 的参数
        model_kwargs = kwargs.copy()
        if 'name' in model_kwargs:
            del model_kwargs['name']
            
        start_time = time.time()
        
        # 使用过滤后的 model_kwargs 调用 C++ 函数
        tokens, profile = self.model.generate_with_profiling(
            prompt=prompt,
            mask_token_id=mask_token_id,
            gen_length=gen_length,
            block_length=block_length,
            denoising_steps=denoising_steps,
            **model_kwargs 
        )
        
        end_time = time.time()
        total_wall_time = (end_time - start_time) * 1000  # Convert to ms
        
        return {
            'tokens': tokens,
            'profile': profile,
            'wall_time_ms': total_wall_time,
            'config': {
                'gen_length': gen_length,
                'block_length': block_length,
                'denoising_steps': denoising_steps,
                **kwargs
            }
        }
    
    def run_comparative_test(
        self,
        prompt: List[int],
        mask_token_id: int,
        configs: List[Dict],
        warmup_before_test: bool = True,
        runs_per_config: int = 1  # 支持多次运行取平均
    ):
        """运行多配置对比测试"""
        
        # 在所有测试前进行warmup
        if warmup_before_test:
            self.warmup(prompt, mask_token_id)
        
        results = []
        
        for i, config in enumerate(configs):
            config_name = config.get('name', f'Config {i+1}')
            print(f"\n{'='*80}")
            print(f"Running test {i+1}/{len(configs)}: {config_name}")
            print(f"{'='*80}")
            
            # 多次运行支持
            run_results = []
            for run_idx in range(runs_per_config):
                if runs_per_config > 1:
                    print(f"\nRun {run_idx + 1}/{runs_per_config}:")
                
                result = self.run_profiling_test(
                    prompt=prompt,
                    mask_token_id=mask_token_id,
                    ensure_warmup=False,  # 已经warmup过了
                    **config
                )
                run_results.append(result)
                
                # 运行间短暂等待
                if run_idx < runs_per_config - 1:
                    time.sleep(0.5)
            
            # 如果多次运行，计算平均值
            if runs_per_config > 1:
                averaged_result = self._average_results(run_results)
                averaged_result['config_name'] = config_name
                averaged_result['individual_runs'] = run_results
                results.append(averaged_result)
                print(f"\nAverage across {runs_per_config} runs:")
                self._print_result_summary(averaged_result)
            else:
                result = run_results[0]
                result['config_name'] = config_name
                results.append(result)
                self._print_result_summary(result)
        
        return results
    
    def _average_results(self, run_results: List[Dict]) -> Dict:
        """计算多次运行的平均结果"""
        avg_result = {
            'tokens': run_results[0]['tokens'],  # 使用第一次的tokens
            'wall_time_ms': sum(r['wall_time_ms'] for r in run_results) / len(run_results),
            'config': run_results[0]['config'],
            'profile': {}
        }
        
        # 平均profile数据
        all_sections = set()
        for result in run_results:
            all_sections.update(result['profile'].keys())
        
        for section in all_sections:
            total_ms_values = []
            avg_ms_values = []
            call_count_values = []
            
            for result in run_results:
                if section in result['profile']:
                    stats = result['profile'][section]
                    total_ms_values.append(stats.get('total_ms', 0))
                    avg_ms_values.append(stats.get('avg_ms', 0))
                    call_count_values.append(stats.get('call_count', 0))
            
            if total_ms_values:
                avg_result['profile'][section] = {
                    'total_ms': sum(total_ms_values) / len(total_ms_values),
                    'avg_ms': sum(avg_ms_values) / len(avg_ms_values),
                    'call_count': sum(call_count_values) / len(call_count_values)
                }
        
        return avg_result
    
    def _print_result_summary(self, result: Dict):
        """打印单次测试摘要"""
        profile = result['profile']
        
        print(f"\nWall Time: {result['wall_time_ms']:.2f} ms")
        print(f"Tokens Generated: {len(result['tokens'])}")
        print(f"Throughput: {len(result['tokens']) / (result['wall_time_ms'] / 1000):.2f} tokens/sec")
        
        if 'total_generation' in profile:
            total_gen = profile['total_generation']
            print(f"Total Generation Time: {total_gen.get('total_ms', 0):.2f} ms")
        
        # Top bottlenecks
        sorted_sections = sorted(
            profile.items(),
            key=lambda x: x[1].get('total_ms', 0),
            reverse=True
        )[:10]
        
        print("\nTop 10 Time-Consuming Sections:")
        print(f"{'Section':<40} {'Total (ms)':<12} {'Avg (ms)':<12} {'Calls':<8}")
        print("-" * 80)
        
        for section, stats in sorted_sections:
            print(f"{section:<40} {stats.get('total_ms', 0):<12.2f} "
                  f"{stats.get('avg_ms', 0):<12.2f} {int(stats.get('call_count', 0)):<8}")
    
    def analyze_bottlenecks(self, result: Dict):
        """分析性能瓶颈"""
        profile = result['profile']
        total_time = result['wall_time_ms']
        
        analysis = {
            'total_time_ms': total_time,
            'breakdown': {},
            'bottlenecks': []
        }
        
        # Calculate percentage breakdown
        for section, stats in profile.items():
            section_total = stats.get('total_ms', 0)
            percentage = (section_total / total_time * 100) if total_time > 0 else 0
            
            analysis['breakdown'][section] = {
                'time_ms': section_total,
                'percentage': percentage,
                'avg_ms': stats.get('avg_ms', 0),
                'calls': int(stats.get('call_count', 0))
            }
            
            # Identify bottlenecks (>10% of total time or >50ms average)
            if percentage > 10 or stats.get('avg_ms', 0) > 50:
                analysis['bottlenecks'].append({
                    'section': section,
                    'time_ms': section_total,
                    'percentage': percentage,
                    'avg_ms': stats.get('avg_ms', 0)
                })
        
        # Sort bottlenecks by percentage
        analysis['bottlenecks'].sort(key=lambda x: x['percentage'], reverse=True)
        
        return analysis
    
    def print_bottleneck_analysis(self, result: Dict):
        """打印瓶颈分析"""
        analysis = self.analyze_bottlenecks(result)
        
        print(f"\n{'='*80}")
        print("BOTTLENECK ANALYSIS")
        print(f"{'='*80}\n")
        
        print(f"Total Time: {analysis['total_time_ms']:.2f} ms\n")
        
        if analysis['bottlenecks']:
            print("Critical Bottlenecks (>10% of total time or >50ms avg):")
            print(f"{'Section':<40} {'Time (ms)':<12} {'Percentage':<12} {'Avg (ms)':<12}")
            print("-" * 80)
            
            for bottleneck in analysis['bottlenecks']:
                print(f"{bottleneck['section']:<40} "
                      f"{bottleneck['time_ms']:<12.2f} "
                      f"{bottleneck['percentage']:<12.1f}% "
                      f"{bottleneck['avg_ms']:<12.2f}")
        else:
            print("No critical bottlenecks identified.")
    
    def visualize_profile(self, result: Dict, output_file: str = 'profile_viz.png'):
        """可视化性能分析结果"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        profile = result['profile']
        
        # Filter out very small sections
        filtered_profile = {
            k: v for k, v in profile.items()
            if v.get('total_ms', 0) > 1.0  # Only show sections > 1ms
        }
        
        if not filtered_profile:
            print("No significant sections to visualize")
            return
        
        # Sort by total time
        sorted_sections = sorted(
            filtered_profile.items(),
            key=lambda x: x[1].get('total_ms', 0),
            reverse=True
        )[:15]  # Top 15 sections
        
        sections = [s[0] for s in sorted_sections]
        times = [s[1].get('total_ms', 0) for s in sorted_sections]
  
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart
        y_pos = np.arange(len(sections))
        ax1.barh(y_pos, times)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(sections, fontsize=8)
        ax1.invert_yaxis()
        ax1.set_xlabel('Time (ms)')
        ax1.set_title('Top Time-Consuming Sections')
        ax1.grid(axis='x', alpha=0.3)
        
        # Pie chart
        ax2.pie(times, labels=sections, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Time Distribution')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to {output_file}")
        plt.close()
    
    def export_results(self, results: List[Dict], output_file: str = 'profile_results.json', archive: bool = True):
        """导出结果到JSON，并可选择归档到profile_runs目录"""
        import os
        from datetime import datetime
        import shutil
        
        export_data = []
        
        for result in results:
            data = {
                'config_name': result.get('config_name', 'Unknown'),
                'wall_time_ms': result['wall_time_ms'],
                'tokens_generated': len(result['tokens']),
                'throughput_tokens_per_sec': len(result['tokens']) / (result['wall_time_ms'] / 1000),
                'config': result['config'],
                'profile': result['profile']
            }
            
            # 如果有多次运行的数据
            if 'individual_runs' in result:
                data['individual_runs'] = [
                    {
                        'wall_time_ms': r['wall_time_ms'],
                        'tokens_generated': len(r['tokens'])
                    }
                    for r in result['individual_runs']
                ]
            
            export_data.append(data)
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\nResults exported to {output_file}")
        
        if archive:
            archive_dir = 'profile_runs'
            if not os.path.exists(archive_dir):
                os.makedirs(archive_dir)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_subdir = os.path.join(archive_dir, timestamp)
            os.makedirs(archive_subdir, exist_ok=True)
            
            # 复制JSON文件
            archive_json = os.path.join(archive_subdir, output_file)
            shutil.copy2(output_file, archive_json)
            
            # 复制所有profile图片
            for i in range(len(results)):
                profile_png = f'profile_{i}.png'
                if os.path.exists(profile_png):
                    shutil.copy2(profile_png, os.path.join(archive_subdir, profile_png))
            
            print(f"Results archived to {archive_subdir}")


def main():
    """示例测试"""
    # 配置
    MODEL_PATH = "/home/lzx/SDAR/training/model/SDAR-1.7B-Chat/SDAR-1.7B-Chat-F16.gguf"
    TOKENIZER_PATH = "/home/lzx/SDAR/training/model/SDAR-1.7B-Chat"
    
    # 加载tokenizer获取真实的prompt
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    MASK_TOKEN_ID = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    
    # 使用真实的prompt
    messages = [{"role": "user", "content": "写一个关于一个机器人第一次发现音乐的短篇故事。"}]
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    
    # 初始化 profiler
    profiler = DiffusionProfiler(MODEL_PATH, n_ctx=8192, n_gpu_layers=35)
    
    # 定义测试配置（仅包含质量过关的配置）
    test_configs = [
        # Baseline配置（用于对比）
        {
            'name': 'Baseline (block=8, steps=8)',
            'gen_length': 128,
            'block_length': 8,
            'denoising_steps': 8,
            'remasking_strategy': 'low_confidence_dynamic'
        },
        {
            'name': 'Baseline (block=4, steps=4)',
            'gen_length': 128,
            'block_length': 4,
            'denoising_steps': 4,
            'remasking_strategy': 'low_confidence_dynamic'
        },
        {
            'name': 'Baseline (block=8, steps=4)',
            'gen_length': 128,
            'block_length': 8,
            'denoising_steps': 4,
            'remasking_strategy': 'low_confidence_dynamic'
        },
        # GPU加速的Baseline
        {
            'name': 'Baseline (block=8, steps=8) + GPU',
            'gen_length': 128,
            'block_length': 8,
            'denoising_steps': 8,
            'remasking_strategy': 'low_confidence_dynamic',
            'use_gpu_sampler': True
        },
        {
            'name': 'Baseline (block=4, steps=4) + GPU',
            'gen_length': 128,
            'block_length': 4,
            'denoising_steps': 4,
            'remasking_strategy': 'low_confidence_dynamic',
            'use_gpu_sampler': True
        },
        {
            'name': 'Baseline (block=8, steps=4) + GPU',
            'gen_length': 128,
            'block_length': 8,
            'denoising_steps': 4,
            'remasking_strategy': 'low_confidence_dynamic',
            'use_gpu_sampler': True
        }
    ]
    
    # 运行对比测试 (自动warmup + 每个配置运行3次取平均)
    results = profiler.run_comparative_test(
        prompt, 
        MASK_TOKEN_ID, 
        test_configs,
        warmup_before_test=True,
        runs_per_config=3  # 每个配置运行3次
    )
    
    # 详细分析每个结果
    for i, result in enumerate(results):
        print(f"\n{'='*80}")
        print(f"Detailed Analysis for: {result['config_name']}")
        print(f"{'='*80}")
        profiler.print_bottleneck_analysis(result)
        profiler.visualize_profile(result, f'profile_{i}.png')
    
    # 导出结果
    profiler.export_results(results)
    
    # 对比总结
    print(f"\n{'='*80}")
    print("COMPARATIVE SUMMARY")
    print(f"{'='*80}\n")
    print(f"{'Config':<30} {'Wall Time (ms)':<15} {'Tokens/sec':<15} {'Speedup':<10}")
    print("-" * 70)
    
    baseline_time = results[0]['wall_time_ms']
    for result in results:
        tokens_per_sec = len(result['tokens']) / (result['wall_time_ms'] / 1000)
        speedup = baseline_time / result['wall_time_ms']
        print(f"{result['config_name']:<30} "
              f"{result['wall_time_ms']:<15.2f} "
              f"{tokens_per_sec:<15.2f} "
              f"{speedup:<10.2f}x")


if __name__ == '__main__':
    main()
