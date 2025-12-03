#!/usr/bin/env python3
import llama_diffusion
from transformers import AutoTokenizer

def main():
    # 配置
    # model_path = "./models/your-diffusion-model.gguf"  # GGUF 格式模型
    # tokenizer_path = "./models/your-diffusion-model"
    model_path = "/home/lzx/SDAR/training/model/SDAR-1.7B-Chat/SDAR-1.7B-Chat-F16.gguf"
    tokenizer_path = "/home/lzx/SDAR/training/model/SDAR-1.7B-Chat"
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    # 获取 mask token ID
    mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    eos_token_id = tokenizer.eos_token_id
    
    # 创建 diffusion 模型
    model = llama_diffusion.LlamaDiffusion(
        model_path=model_path,
        n_ctx=8192,
        n_gpu_layers=35  # 根据你的 GPU 调整
    )
    
    # 准备输入
    # messages = [
    #     {"role": "user", "content": "If the domain of the function $\\log x^2$ is $x < a$ or $x > b$, for some $a$ and $b$, find $a + b$.\\nPlease reason step by step, and put your final answer within \\boxed{}.\\n"}
    # ]
    messages = [
        {
            "role": "user", 
            "content": "Write a short story about a robot who discovers music for the first time."
        }
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=False
    )
    
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    
    # 生成
    output_tokens = model.generate(
        prompt=prompt_tokens,
        mask_token_id=mask_token_id,
        gen_length=2048,
        block_length=1,
        denoising_steps=1,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        remasking_strategy="low_confidence_dynamic",
        confidence_threshold=0.85,
        stop_token_ids=[eos_token_id],
        use_gpu_sampler=True
    )
    
    # generate 返回的序列包含 prompt + 生成的内容
    # 只提取生成的部分（从 prompt_length 开始）
    generated_tokens = output_tokens[len(prompt_tokens):]
    
    # 解码生成的 token
    try:
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    except (TypeError, ValueError) as e:
        # 如果解码失败，尝试逐个 token 解码并过滤 None
        print(f"Warning: Direct decode failed ({e}), trying token-by-token decode...")
        decoded_parts = []
        for token_id in generated_tokens:
            try:
                token_str = tokenizer.decode([token_id], skip_special_tokens=False)
                if token_str is not None:
                    decoded_parts.append(token_str)
            except:
                continue
        generated_text = "".join(decoded_parts)
    
    # 清理 mask token
    if '<|MASK|>' in generated_text:
        generated_text = generated_text.replace('<|MASK|>', '')
    
    # 移除结束标记
    generated_part = generated_text.strip()
    if generated_part.endswith('<|im_end|>'):
        generated_part = generated_part[:-10].strip()
    if generated_part.endswith('<|endoftext|>'):
        generated_part = generated_part[:-13].strip()
    
    print("=" * 80)
    print("Generated Output:")
    print("=" * 80)
    print(generated_part)

if __name__ == "__main__":
    main()
