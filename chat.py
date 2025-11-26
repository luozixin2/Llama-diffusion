#!/usr/bin/env python3
"""
使用 llama_diffusion 进行多轮对话的命令行脚本
"""
import llama_diffusion
from transformers import AutoTokenizer
import sys

class ChatSession:
    def __init__(self, model, tokenizer, mask_token_id, eos_token_id, use_gpu_sampler=False):
        self.model = model
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        self.eos_token_id = eos_token_id
        self.messages = []
        self.use_gpu_sampler = use_gpu_sampler
        
    def stream_callback(self, token_ids: list[int]):
        """解码并打印新生成的 token"""
        text_chunk = self.tokenizer.decode(token_ids)
        if '<|MASK|>' not in text_chunk:
            print(text_chunk, end='', flush=True)
    
    def generate_response(self, user_input: str, gen_length=512, denoising_steps=4, 
                         temperature=0.8, top_p=0.95):
        """生成助手的回复"""
        # 添加用户消息
        self.messages.append({"role": "user", "content": user_input})
        
        # 构建完整的对话历史
        prompt_text = self.tokenizer.apply_chat_template(
            self.messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
        prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        
        print("助手: ", end='', flush=True)
        
        # 用于收集完整的回复
        self.current_response = []
        
        def callback_with_collection(token_ids: list[int]):
            self.current_response.extend(token_ids)
            self.stream_callback(token_ids)
        
        try:
            # 流式生成
            self.model.generate_stream(
                prompt=prompt_tokens,
                callback=callback_with_collection,
                mask_token_id=self.mask_token_id,
                gen_length=gen_length,
                block_length=4,
                denoising_steps=denoising_steps,
                temperature=temperature,
                top_p=top_p,
                stop_token_ids=[self.eos_token_id] if self.eos_token_id is not None else [],
                use_gpu_sampler=self.use_gpu_sampler
            )
            print()  # 换行
            
            # 将助手回复添加到历史中
            response_text = self.tokenizer.decode(self.current_response, skip_special_tokens=True)
            # 清理可能的 mask token
            response_text = response_text.replace('<|MASK|>', '').strip()
            self.messages.append({"role": "assistant", "content": response_text})
            
        except Exception as e:
            print(f"\n生成过程中出错: {e}")
            import traceback
            traceback.print_exc()
            # 如果生成失败，移除最后添加的用户消息
            self.messages.pop()
    
    def clear_history(self):
        """清空对话历史"""
        self.messages = []
        print("对话历史已清空")
    
    def show_history(self):
        """显示对话历史"""
        if not self.messages:
            print("暂无对话历史")
            return
        
        print("\n" + "=" * 80)
        print("对话历史:")
        print("=" * 80)
        for msg in self.messages:
            role = "用户" if msg["role"] == "user" else "助手"
            print(f"{role}: {msg['content']}\n")
        print("=" * 80 + "\n")

def print_help():
    """打印帮助信息"""
    help_text = """
可用命令:
  /help       - 显示此帮助信息
  /clear      - 清空对话历史
  /history    - 显示对话历史
  /settings   - 显示当前设置
  /set <参数> <值> - 修改参数 (gen_length, temperature, top_p, denoising_steps)
  /exit, /quit - 退出程序
  
直接输入文本即可开始对话
"""
    print(help_text)

def main():
    # --- 配置 ---
    model_path = r"F:\DLLm\SDAR-1.7B-Chat-Q6_K.gguf"
    tokenizer_path = r"F:\DLLm\SDAR-1.7B-Chat"
    
    # 生成参数（可通过命令修改）
    settings = {
        'gen_length': 512,
        'temperature': 0.8,
        'top_p': 0.95,
        'denoising_steps': 4
    }
    
    print("=" * 80)
    print("正在加载 Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        eos_token_id = tokenizer.eos_token_id
        print(f"Mask token ID: {mask_token_id}, EOS token ID: {eos_token_id}")
    except Exception as e:
        print(f"加载 Tokenizer 失败: {e}")
        sys.exit(1)
    
    print("=" * 80)
    print("正在加载模型...")
    try:
        model = llama_diffusion.LlamaDiffusion(
            model_path=model_path,
            n_ctx=8192,
            n_gpu_layers=35  # 根据你的 GPU 调整
        )
        print("模型加载成功!")
    except Exception as e:
        print(f"加载模型失败: {e}")
        sys.exit(1)
    
    # 创建对话会话
    session = ChatSession(model, tokenizer, mask_token_id, eos_token_id, use_gpu_sampler=True)
    
    print("=" * 80)
    print("多轮对话系统已启动!")
    print("输入 /help 查看可用命令")
    print("=" * 80)
    
    # 主循环
    while True:
        try:
            # 获取用户输入
            user_input = input("\n用户: ").strip()
            
            # 处理空输入
            if not user_input:
                continue
            
            # 处理命令
            if user_input.startswith('/'):
                cmd_parts = user_input.split()
                cmd = cmd_parts[0].lower()
                
                if cmd in ['/exit', '/quit']:
                    print("再见!")
                    break
                
                elif cmd == '/help':
                    print_help()
                
                elif cmd == '/clear':
                    session.clear_history()
                
                elif cmd == '/history':
                    session.show_history()
                
                elif cmd == '/settings':
                    print("\n当前设置:")
                    for key, value in settings.items():
                        print(f"  {key}: {value}")
                    print()
                
                elif cmd == '/set':
                    if len(cmd_parts) != 3:
                        print("用法: /set <参数名> <值>")
                        print("例如: /set temperature 0.7")
                        continue
                    
                    param = cmd_parts[1]
                    if param not in settings:
                        print(f"未知参数: {param}")
                        print(f"可用参数: {', '.join(settings.keys())}")
                        continue
                    
                    try:
                        value = float(cmd_parts[2]) if param != 'gen_length' and param != 'denoising_steps' else int(cmd_parts[2])
                        settings[param] = value
                        print(f"已设置 {param} = {value}")
                    except ValueError:
                        print("无效的值")
                
                else:
                    print(f"未知命令: {cmd}，输入 /help 查看帮助")
                
                continue
            
            # 生成回复
            session.generate_response(
                user_input,
                gen_length=settings['gen_length'],
                temperature=settings['temperature'],
                top_p=settings['top_p'],
                denoising_steps=settings['denoising_steps']
            )
            
        except KeyboardInterrupt:
            print("\n\n检测到中断，输入 /exit 退出程序")
            continue
        except EOFError:
            print("\n再见!")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
