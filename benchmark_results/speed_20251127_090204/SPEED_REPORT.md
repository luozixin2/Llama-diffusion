# SDAR 1.7B Speed Benchmark Report

**测试时间**: 2025-11-27 09:02:04

## 测试配置
- 模型: SDAR-1.7B-Chat (F16 GGUF)
- 生成长度: 256 tokens
- 测试次数: 10
- 预热次数: 2
- 温度: 1.0, top_p: 1.0, top_k: 0

## 速度测试结果

| 配置 | Block Length | Denoising Steps | GPU Sampler | 平均时间(s) | 吞吐量(tokens/s) |
|------|--------------|-----------------|-------------|------------|-----------------|
| Block1_Step1_GPU | 1 | 1 | ✅ | 4.127 | 62.03 |
| Block2_Step2_GPU | 2 | 2 | ✅ | 2.678 | 95.6 |
| Block3_Step3_GPU | 3 | 3 | ✅ | 3.986 | 64.22 |
| Block4_Step4_GPU | 4 | 4 | ✅ | 2.984 | 85.79 |
| Block4_Step4_CPU | 4 | 4 | ❌ | 5.645 | 45.35 |

## GPU Sampler vs CPU Sampler 对比 (Block4_Step4)

| 指标 | GPU Sampler | CPU Sampler | 差异 |
|------|-------------|-------------|------|
| 平均时间 | 2.984s | 5.645s | 1.89x |
| 吞吐量 | 85.79 t/s | 45.35 t/s | - |

## 结论

- Block Length和Denoising Steps增加会提高质量但降低速度
- Block1_Step1 最快但质量最低
- Block4_Step4 是官方推荐配置，平衡质量和速度
