# SDAR 1.7B 综合基准测试报告

**测试时间**: 2025-11-27 06:24:10

## 测试配置

| 配置名称 | Block Length | Denoising Steps | GPU Sampler |
|----------|--------------|-----------------|-------------|
| Block1_Step1_GPU | 1 | 1 | ✅ |
| Block2_Step2_GPU | 2 | 2 | ✅ |
| Block3_Step3_GPU | 3 | 3 | ✅ |
| Block4_Step4_GPU | 4 | 4 | ✅ |
| Block4_Step4_CPU | 4 | 4 | ❌ |

## 准确率结果

| Dataset | Block1_Step1_GPU | Block2_Step2_GPU | Block3_Step3_GPU | Block4_Step4_GPU | Block4_Step4_CPU |
|---------|--------|--------|--------|--------|--------|
| GSM8K | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

## 性能结果 (Tokens/sec)

| Dataset | Block1_Step1_GPU | Block2_Step2_GPU | Block3_Step3_GPU | Block4_Step4_GPU | Block4_Step4_CPU |
|---------|--------|--------|--------|--------|--------|
| GSM8K | 68.6 | 22.8 | 20.2 | 20.7 | 23.5 |

## 平均延迟 (ms/sample)

| Dataset | Block1_Step1_GPU | Block2_Step2_GPU | Block3_Step3_GPU | Block4_Step4_GPU | Block4_Step4_CPU |
|---------|--------|--------|--------|--------|--------|
| GSM8K | 7458 | 22455 | 25342 | 24703 | 21788 |

## GPU Sampler vs CPU Sampler 对比 (Block4_Step4)

| Dataset | GPU Throughput | CPU Throughput | Speedup |
|---------|---------------|----------------|----------|
| GSM8K | 20.7 | 23.5 | 0.88x |

## 详细结果

### GSM8K - Block1_Step1_GPU
- **准确率**: 0.00% (0/3)
- **总耗时**: 22.4s
- **平均延迟**: 7458ms/sample
- **吞吐量**: 68.6 tokens/sec
- **生成tokens**: 1536

### GSM8K - Block2_Step2_GPU
- **准确率**: 0.00% (0/3)
- **总耗时**: 67.4s
- **平均延迟**: 22455ms/sample
- **吞吐量**: 22.8 tokens/sec
- **生成tokens**: 1536

### GSM8K - Block3_Step3_GPU
- **准确率**: 0.00% (0/3)
- **总耗时**: 76.0s
- **平均延迟**: 25342ms/sample
- **吞吐量**: 20.2 tokens/sec
- **生成tokens**: 1536

### GSM8K - Block4_Step4_GPU
- **准确率**: 0.00% (0/3)
- **总耗时**: 74.1s
- **平均延迟**: 24703ms/sample
- **吞吐量**: 20.7 tokens/sec
- **生成tokens**: 1536

### GSM8K - Block4_Step4_CPU
- **准确率**: 0.00% (0/3)
- **总耗时**: 65.4s
- **平均延迟**: 21788ms/sample
- **吞吐量**: 23.5 tokens/sec
- **生成tokens**: 1536

