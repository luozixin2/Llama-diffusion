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
        """可视化性能分析结果 - 确保不互相包含，多次调用函数单独展示"""
        import matplotlib.pyplot as plt
        import numpy as np
        import re
        
        profile = result['profile']
        total_wall_time = result['wall_time_ms']
        
        # Filter out very small sections
        filtered_profile = {
            k: v for k, v in profile.items()
            if v.get('total_ms', 0) > 1.0  # Only show sections > 1ms
        }
        
        if not filtered_profile:
            print("No significant sections to visualize")
            return
        
        # 分类阶段
        top_level_phases = ['prefill_phase', 'generation_phase']
        block_pattern = re.compile(r'^block_\d+$')
        denoising_step_pattern = re.compile(r'^denoising_step_\d+$')
        
        # 识别不同类型的阶段
        top_level_data = {}  # 顶层阶段
        block_data = {}  # block_X 阶段
        denoising_step_data = {}  # denoising_step_X 阶段
        multi_call_functions = {}  # 多次调用的函数 (call_count > 1)
        other_sections = {}  # 其他单次调用的阶段
        
        for name, stats in filtered_profile.items():
            call_count = stats.get('call_count', 0)
            total_ms = stats.get('total_ms', 0)
            
            if name in top_level_phases:
                top_level_data[name] = stats
            elif block_pattern.match(name):
                block_data[name] = stats
            elif denoising_step_pattern.match(name):
                denoising_step_data[name] = stats
            elif call_count > 1:
                multi_call_functions[name] = stats
            else:
                other_sections[name] = stats
        
        # 计算顶层阶段的"其他时间"（不包含子阶段的时间）
        # generation_phase 包含所有 block_X，block_X 包含 denoising_step_X
        # prefill_phase 可能包含 prefill_llama_decode 等
        
        # 计算 generation_phase 的子阶段总时间
        generation_phase_stats = top_level_data.get('generation_phase')
        if generation_phase_stats:
            # 计算所有 block 的总时间
            blocks_total_time = sum(s.get('total_ms', 0) for s in block_data.values())
            # generation_phase 的其他时间 = generation_phase 总时间 - 所有 block 时间
            generation_other_time = max(0, generation_phase_stats.get('total_ms', 0) - blocks_total_time)
        else:
            generation_other_time = 0
        
        # 计算 prefill_phase 的子阶段总时间
        prefill_phase_stats = top_level_data.get('prefill_phase')
        prefill_sub_sections = {k: v for k, v in filtered_profile.items() 
                               if k.startswith('prefill_') and k != 'prefill_phase'}
        prefill_sub_total = sum(s.get('total_ms', 0) for s in prefill_sub_sections.values())
        if prefill_phase_stats:
            prefill_other_time = max(0, prefill_phase_stats.get('total_ms', 0) - prefill_sub_total)
        else:
            prefill_other_time = 0
        
        # 计算每个 block 的子阶段（denoising_step）总时间
        # 需要将 denoising_step 按 block 分组（这需要从调用关系推断，这里简化处理）
        # 由于 denoising_step 是在 block 内调用的，我们计算所有 denoising_step 的总时间
        denoising_steps_total = sum(s.get('total_ms', 0) for s in denoising_step_data.values())
        
        # 构建饼图数据 - 确保不互相包含
        pie_labels = []
        pie_values = []
        pie_colors = []
        
        # 1. Top-level phases other time (excluding sub-phases)
        if generation_other_time > 1.0:
            pie_labels.append('generation_phase (other)')
            pie_values.append(generation_other_time)
            pie_colors.append('#FF6B6B')
        
        if prefill_other_time > 1.0:
            pie_labels.append('prefill_phase (other)')
            pie_values.append(prefill_other_time)
            pie_colors.append('#4ECDC4')
        
        # 2. All blocks total time (subtract denoising_step time to avoid duplication)
        # Simplified: show block total time, but not denoising_step (they are included in blocks)
        blocks_total = sum(s.get('total_ms', 0) for s in block_data.values())
        blocks_net_time = max(0, blocks_total - denoising_steps_total)
        if blocks_net_time > 1.0:
            pie_labels.append('blocks (excl. denoising)')
            pie_values.append(blocks_net_time)
            pie_colors.append('#95E1D3')
        
        # 3. denoising_step total time
        if denoising_steps_total > 1.0:
            pie_labels.append('denoising_steps (total)')
            pie_values.append(denoising_steps_total)
            pie_colors.append('#F38181')
        
        # 4. Multi-call functions (show total time separately)
        for name, stats in sorted(multi_call_functions.items(), 
                                 key=lambda x: x[1].get('total_ms', 0), reverse=True):
            total_ms = stats.get('total_ms', 0)
            call_count = int(stats.get('call_count', 0))
            if total_ms > 1.0:
                pie_labels.append(f'{name}\n({call_count} calls)')
                pie_values.append(total_ms)
                pie_colors.append('#AA96DA')
        
        # 5. Other single-call important phases
        for name, stats in sorted(other_sections.items(), 
                                 key=lambda x: x[1].get('total_ms', 0), reverse=True)[:5]:
            total_ms = stats.get('total_ms', 0)
            if total_ms > 1.0:
                pie_labels.append(name)
                pie_values.append(total_ms)
                pie_colors.append('#FCBAD3')
        
        # Calculate unclassified time
        accounted_time = sum(pie_values)
        unaccounted_time = max(0, total_wall_time - accounted_time)
        if unaccounted_time > 1.0:
            pie_labels.append('Other/Unclassified')
            pie_values.append(unaccounted_time)
            pie_colors.append('#C7C7C7')
        
        # Build bar chart data - categorized display
        bar_categories = {
            'Top-level Phases': [],
            'Block Phases': [],
            'Denoising Steps': [],
            'Multi-call Functions': [],
            'Other Phases': []
        }
        
        # Top-level phases
        if generation_other_time > 1.0:
            bar_categories['Top-level Phases'].append(('generation_phase (other)', generation_other_time))
        if prefill_other_time > 1.0:
            bar_categories['Top-level Phases'].append(('prefill_phase (other)', prefill_other_time))
        
        # Block phases (show top 10)
        for name, stats in sorted(block_data.items(), 
                                 key=lambda x: x[1].get('total_ms', 0), reverse=True)[:10]:
            total_ms = stats.get('total_ms', 0)
            if total_ms > 1.0:
                bar_categories['Block Phases'].append((name, total_ms))
        
        # Denoising steps (show top 10)
        for name, stats in sorted(denoising_step_data.items(), 
                                 key=lambda x: x[1].get('total_ms', 0), reverse=True)[:10]:
            total_ms = stats.get('total_ms', 0)
            if total_ms > 1.0:
                call_count = int(stats.get('call_count', 0))
                bar_categories['Denoising Steps'].append((f'{name} ({call_count} calls)', total_ms))
        
        # Multi-call functions
        for name, stats in sorted(multi_call_functions.items(), 
                                 key=lambda x: x[1].get('total_ms', 0), reverse=True)[:10]:
            total_ms = stats.get('total_ms', 0)
            call_count = int(stats.get('call_count', 0))
            if total_ms > 1.0:
                bar_categories['Multi-call Functions'].append((f'{name} ({call_count} calls)', total_ms))
        
        # Other phases
        for name, stats in sorted(other_sections.items(), 
                                 key=lambda x: x[1].get('total_ms', 0), reverse=True)[:10]:
            total_ms = stats.get('total_ms', 0)
            if total_ms > 1.0:
                bar_categories['Other Phases'].append((name, total_ms))
        
        # 创建图表
        fig = plt.figure(figsize=(20, 10))
        
        # 饼图 - 左侧
        ax1 = plt.subplot(2, 2, 1)
        if pie_values:
            # 只显示占比大于1%的项
            significant_indices = [i for i, v in enumerate(pie_values) 
                                 if v / sum(pie_values) * 100 > 1.0]
            if significant_indices:
                filtered_labels = [pie_labels[i] for i in significant_indices]
                filtered_values = [pie_values[i] for i in significant_indices]
                filtered_colors = [pie_colors[i] for i in significant_indices]
                
                wedges, texts, autotexts = ax1.pie(
                    filtered_values, 
                    labels=filtered_labels, 
                    autopct='%1.1f%%', 
                    startangle=90,
                    colors=filtered_colors,
                    textprops={'fontsize': 8}
                )
                # Adjust percentage text size
                for autotext in autotexts:
                    autotext.set_fontsize(7)
            else:
                ax1.text(0.5, 0.5, 'No significant data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Time Distribution (Non-overlapping)', fontsize=12, fontweight='bold')
        
        # Multi-call functions ratio - top right
        ax2 = plt.subplot(2, 2, 2)
        multi_call_total = sum(s.get('total_ms', 0) for s in multi_call_functions.values())
        multi_call_percentage = (multi_call_total / total_wall_time * 100) if total_wall_time > 0 else 0
        other_time = max(0, total_wall_time - multi_call_total)
        other_percentage = 100 - multi_call_percentage
        
        if multi_call_total > 1.0 and other_time > 0:
            ax2.pie(
                [multi_call_total, other_time],
                labels=[f'Multi-call Functions\n({multi_call_percentage:.1f}%)', 
                       f'Other Time\n({other_percentage:.1f}%)'],
                autopct='%1.1f%%',
                startangle=90,
                colors=['#AA96DA', '#E0E0E0'],
                textprops={'fontsize': 9}
            )
        elif multi_call_total > 1.0:
            # If multi-call functions take all time
            ax2.pie(
                [multi_call_total],
                labels=[f'Multi-call Functions\n({multi_call_percentage:.1f}%)'],
                autopct='%1.1f%%',
                startangle=90,
                colors=['#AA96DA'],
                textprops={'fontsize': 9}
            )
        else:
            ax2.text(0.5, 0.5, 'No multi-call function data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Multi-call Functions Time Ratio', fontsize=12, fontweight='bold')
        
        # Bar chart - bottom (categorized display)
        ax3 = plt.subplot(2, 1, 2)
        
        all_bar_labels = []
        all_bar_values = []
        category_colors = {
            'Top-level Phases': '#FF6B6B',
            'Block Phases': '#95E1D3',
            'Denoising Steps': '#F38181',
            'Multi-call Functions': '#AA96DA',
            'Other Phases': '#FCBAD3'
        }
        bar_colors_list = []
        
        y_offset = 0
        for category, items in bar_categories.items():
            if items:
                for label, value in items:
                    all_bar_labels.append(label)
                    all_bar_values.append(value)
                    bar_colors_list.append(category_colors[category])
                y_offset += len(items) + 1  # Add separator
        
        if all_bar_values:
            y_pos = np.arange(len(all_bar_labels))
            bars = ax3.barh(y_pos, all_bar_values, color=bar_colors_list)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(all_bar_labels, fontsize=7)
            ax3.invert_yaxis()
            ax3.set_xlabel('Time (ms)', fontsize=10)
            ax3.set_title('Phase Time Breakdown (Categorized)', fontsize=12, fontweight='bold')
            ax3.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, all_bar_values)):
                width = bar.get_width()
                ax3.text(width, bar.get_y() + bar.get_height()/2, 
                        f'{value:.1f}ms', 
                        ha='left', va='center', fontsize=6)
        
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
