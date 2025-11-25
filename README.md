# Llama.cpp Diffusion Language Model Support

支持在 llama.cpp 中运行扩散语言模型。

## 编译步骤

### 1. 安装依赖

```bash
# 安装 llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_CUBLAS=ON  # 或其他后端
make -j
sudo make install

# 安装 pybind11
pip install pybind11
2. 编译 Diffusion 扩展
bash
复制代码
mkdir build && cd build
cmake .. -DLLAMA_CUBLAS=ON
make -j
3. 安装 Python 包
bash
复制代码
pip install -e .
使用方法
1. 转换模型
bash
复制代码
python convert_to_gguf.py \
    --model_path /path/to/hf/model \
    --output_path ./models/diffusion-model.gguf
2. 量化模型 (可选)
bash
复制代码
# 使用 llama.cpp 的量化工具
./llama-quantize ./models/diffusion-model.gguf \
                 ./models/diffusion-model-q4_0.gguf q4_0
3. 运行推理
python
复制代码
import llama_diffusion
from transformers import AutoTokenizer

# 加载模型
model = llama_diffusion.LlamaDiffusion(
    model_path="./models/diffusion-model-q4_0.gguf",
    n_ctx=8192,
    n_gpu_layers=35
)

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("path/to/tokenizer")

# 生成
prompt_tokens = tokenizer.encode("Your prompt here")
output_tokens = model.generate(
    prompt=prompt_tokens,
    mask_token_id=tokenizer.mask_token_id,
    gen_length=1024,
    block_length=4,
    denoising_steps=4
)

# 解码
output_text = tokenizer.decode(output_tokens)
print(output_text)

## 使用流程总结

1. **编译 llama.cpp 和扩展**
2. **转换模型**: HF格式 → GGUF格式
3. **量化模型** (可选): 减少内存占用
4. **运行推理**: 使用 Python 接口

