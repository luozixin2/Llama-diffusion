# SDAR Llama-Diffusion 性能优化报告

**日期**: 2025-11-27  
**版本**: v1.1 (优化后)

## 优化总结

| 优化阶段 | 状态 | 效果 |
|---------|------|------|
| Phase 1: Context池化与复用 | ✅ 完成 | **2-3x 加速** (重复调用) |
| Phase 2: KV Cache管理优化 | ✅ 完成 | 包含在Phase 1中 |
| Phase 3: 内存拷贝优化 | ❌ 取消 | 与Phase 1冲突 |
| Phase 4: Android编译支持 | ✅ 完成 | ARM64 NEON优化 |

## 性能测试结果

### Context复用效果 (gen_length=32, block=4, steps=4)

| 调用 | 延迟 | 吞吐量 |
|-----|------|--------|
| 首次调用 (创建context) | 0.369s | 87 tok/s |
| 后续调用 (复用context) | 0.174s | **184 tok/s** |
| **加速比** | | **2.12x** |

### 短序列测试 (gen_length=64)

| 调用 | 延迟 | 吞吐量 |
|-----|------|--------|
| 首次调用 | 0.197s | 324 tok/s |
| 后续调用 | 0.061s | **1050+ tok/s** |
| **加速比** | | **3.21x** |

### 不同配置性能对比 (gen_length=128)

| 配置 | Block | Steps | 延迟 | 吞吐量 |
|-----|-------|-------|------|--------|
| Block1_Step1 | 1 | 1 | 5.09s | 25.1 tok/s |
| Block2_Step2 | 2 | 2 | 4.27s | 30.0 tok/s |
| Block4_Step4 | 4 | 4 | 3.68s | **34.8 tok/s** |

## 核心优化细节

### Phase 1: Context池化与复用

**问题**: 每次`generate()`调用都重新创建`llama_context`，开销巨大。

**解决方案**: 在`LlamaDiffusionWrapper`中缓存context：

```cpp
class LlamaDiffusionWrapper {
private:
    llama_context* cached_ctx_ = nullptr;
    int cached_block_length_ = 0;
    
public:
    llama_context* get_or_create_context(int block_length) {
        if (cached_ctx_ && cached_block_length_ == block_length) {
            // 复用context，只清除KV cache
            llama_memory_t memory = llama_get_memory(cached_ctx_);
            llama_memory_clear(memory, true);
            return cached_ctx_;
        }
        // block_length变化时才重建context
        // ...
    }
};
```

### Phase 4: Android编译支持

新增Android平台编译配置：

- ARM64-v8a: 自动启用NEON优化
- ARMv7a: 启用 `-mfpu=neon` 标志
- 支持Vulkan GPU加速（可选）
- 提供 `build_android.sh` 构建脚本
- 提供 `android_jni.h` C接口头文件

**使用方法**:
```bash
export ANDROID_NDK=/path/to/ndk
./build_android.sh arm64-v8a Release
```

## 文件变更

| 文件 | 变更 |
|------|------|
| `llama_diffusion/diffusion_binding.cpp` | 添加context缓存逻辑 |
| `CMakeLists.txt` | 添加Android平台检测和NEON优化 |
| `build_android.sh` | 新增Android构建脚本 |
| `llama_diffusion/android_jni.h` | 新增C接口头文件 |

## 后续优化建议

1. **多GPU并行**: 当前仅使用单GPU，可探索Tensor Parallelism
2. **量化模型**: 使用Q4_K_M量化进一步加速移动端推理
3. **批处理优化**: 支持多请求批处理提升吞吐
4. **Speculative Decoding**: 使用小模型加速大模型推理

## 兼容性

- ✅ Linux (x86_64) + CUDA
- ✅ Android (arm64-v8a, armeabi-v7a)
- ⚠️ macOS (需测试Metal支持)
- ⚠️ Windows (需测试CUDA支持)

