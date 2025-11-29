#!/usr/bin/env python3
"""
使用 llama_diffusion 进行流式推理的示例脚本
"""
from __future__ import annotations

import llama_diffusion
from transformers import AutoTokenizer
import sys

def main():
    # --- 配置 ---
    #model_path = "./models/your-diffusion-model-q4_0.gguf"  # 量化后的 GGUF 模型
    #tokenizer_path = "./models/your-diffusion-model"       # HF tokenizer 路径
    model_path = "/home/lzx/SDAR/training/model/SDAR-1.7B-Chat/SDAR-1.7B-Chat-F16.gguf"
    tokenizer_path = "/home/lzx/SDAR/training/model/SDAR-1.7B-Chat"
    print("=" * 80)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    # 获取 mask 和 eos token ID
    mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    eos_token_id = tokenizer.eos_token_id
    print(f"Mask token: {mask_token_id}, EOS token: {eos_token_id}")
    
    print("=" * 80)
    print("Loading model...")
    try:
        model = llama_diffusion.LlamaDiffusion(
            model_path=model_path,
            n_ctx=8192,
            n_gpu_layers=35  # 根据你的 GPU 调整
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # --- 准备输入 ---
    messages = [
        # {
        #     "role": "user", 
        #     "content": "Write a short story about a robot who discovers music for the first time."
        # }
        {
            "role": "user", 
            "content": "写一个关于一个机器人第一次发现音乐的短篇故事。"
        }
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=False
    )
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    
    # --- 定义回调函数 ---
    # 这个函数将在每次 C++ 代码生成新 token 块时被调用
    def stream_callback(token_ids: list[int]):
        """Decodes and prints new tokens to the console."""
        # 解码新收到的 token 块
        text_chunk = tokenizer.decode(token_ids)
        # 只有当文本块中不包含 <|MASK|> 时才打印
        if '<|MASK|>' not in text_chunk:
            # 立即打印，不带换行符，并刷新缓冲区
            print(text_chunk, end='', flush=True)

    # --- 开始流式生成 ---
    print("=" * 80)
    print("Prompt:\n", prompt_text)
    print("=" * 80)
    print("Streaming Output:")
    print("-" * 20)
    
    try:
        # 调用 generate_stream，它没有返回值
        model.generate_stream(
            prompt=prompt_tokens,
            callback=stream_callback, # 传入回调函数
            mask_token_id=mask_token_id,
            gen_length=512,
            block_length=4,
            denoising_steps=4,  # 去噪步数
            temperature=0.95,  # 更高的temperature
            top_p=0.95,  # 标准top_p
            repetition_penalty=1.05,  # 使用repetition_penalty抑制重复输出（>1.0降低重复概率，值越大抑制越强）
            stop_token_ids=[eos_token_id] if eos_token_id is not None else [],
            remasking_strategy="low_confidence_dynamic",  # 使用动态低置信度策略
            confidence_threshold=0.85,  # 标准置信度阈值
            use_gpu_sampler=True
        )
        print("\n" + "-" * 20)
        print("Generation finished.")
        
    except Exception as e:
        print(f"\nError during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
