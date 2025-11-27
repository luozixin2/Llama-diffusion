cmake_minimum_required(VERSION 3.18)

# 设置 MSVC 运行时（Windows专用，Linux下忽略但保留无妨）
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# ==========================================
# Android 平台检测与配置
# ==========================================
if(ANDROID)
    message(STATUS "Building for Android: ${ANDROID_ABI}")
    
    # NEON 优化 (ARM64 默认支持)
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        # ARM64 NEON is always available
        add_compile_definitions(GGML_USE_NEON)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -ffast-math")
        message(STATUS "Enabled ARM64 NEON optimizations")
    elseif(ANDROID_ABI STREQUAL "armeabi-v7a")
        # ARMv7 with NEON
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon -O3 -ffast-math")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfpu=neon -O3 -ffast-math")
        add_compile_definitions(GGML_USE_NEON)
        message(STATUS "Enabled ARMv7 NEON optimizations")
    elseif(ANDROID_ABI STREQUAL "x86_64" OR ANDROID_ABI STREQUAL "x86")
        # x86 Android (emulator) - enable AVX if available
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -ffast-math")
        message(STATUS "Building for Android x86/x86_64")
    endif()
    
    # 禁用 CUDA (Android 不支持)
    set(GGML_CUDA OFF CACHE BOOL "Disable CUDA for Android" FORCE)
    
    # 可选: 启用 Vulkan GPU 加速 (需要 Vulkan SDK)
    option(GGML_VULKAN "Enable Vulkan GPU acceleration for Android" OFF)
    if(GGML_VULKAN)
        message(STATUS "Vulkan GPU acceleration enabled")
    endif()
    
    # Android 特定优化
    add_compile_definitions(
        GGML_USE_ANDROID
        _GNU_SOURCE
    )
    
    # 减少二进制大小
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffunction-sections -fdata-sections")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections")
    
    # 不构建 Python 绑定
    set(BUILD_PYTHON_BINDINGS OFF CACHE BOOL "Disable Python bindings for Android" FORCE)
else()
    # 非 Android 平台，使用 CUDA
    set(MY_CUDA_ROOT "/home/lzx/cuda-12.8")
endif()

# ==========================================
# CUDA 配置 (仅非 Android 平台)
# ==========================================
if(NOT ANDROID)

    # 1. 告诉 CMake 哪里找 CUDA 工具包 (用于 find_package)
    set(CUDAToolkit_ROOT "${MY_CUDA_ROOT}" CACHE PATH "CUDA Toolkit Root" FORCE)

    # 2. 强制指定 nvcc 编译器路径
    set(CMAKE_CUDA_COMPILER "${MY_CUDA_ROOT}/bin/nvcc")

    # 3. 辅助变量，防止 CMake 查找系统默认路径
    set(CUDA_TOOLKIT_ROOT_DIR "${MY_CUDA_ROOT}" CACHE PATH "Legacy CUDA Root" FORCE)
endif()

project(SDAR_LlamaDiffusionProject LANGUAGES C CXX)

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==========================================
# 全局开启 -fPIC
# ==========================================
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 强制静态链接子模块，方便打包进 Python 扩展
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)

# GPU 选项
option(LLAMA_USE_ACCELERATE "Use Apple Accelerate (macOS)" OFF)
if(NOT ANDROID)
    option(GGML_CUDA "Use cuBLAS (NVIDIA GPUs)" ON)
endif()
option(LLAMA_METAL "Use Apple Metal (macOS)" OFF)

# 如果开启了 CUDA，显式启用 CUDA 语言支持
if(GGML_CUDA AND NOT ANDROID)
    enable_language(CUDA)
endif()

# ---------------------------------------------------
# 1) extern/pybind11
add_subdirectory(extern/pybind11)

# 2) extern/llama.cpp
# 防止 llama.cpp 构建测试和示例，加快编译速度
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)

add_subdirectory(extern/llama.cpp)

# ==========================================
# 双重保险：强制 llama 目标使用 PIC
# ==========================================
if(TARGET llama)
    set_property(TARGET llama PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()
if(TARGET ggml) 
    set_property(TARGET ggml PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()
if(TARGET ggml-base) # 新版 llama.cpp 可能还有这个
    set_property(TARGET ggml-base PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()
if(TARGET ggml-cuda) # 如果有独立的 cuda 目标
    set_property(TARGET ggml-cuda PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

# 3) llama_diffusion
add_subdirectory(llama_diffusion)
