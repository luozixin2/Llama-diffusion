cmake_minimum_required(VERSION 3.18)
project(llama_diffusion LANGUAGES C CXX)

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# llama.cpp 路径
set(LLAMA_ROOT    ${CMAKE_SOURCE_DIR}/extern/llama.cpp)
set(LLAMA_INCLUDE ${LLAMA_ROOT})
set(LLAMA_SRC     ${LLAMA_ROOT}/src)
set(LLAMA_COMMON  ${LLAMA_ROOT}/common)

# =======================================================
# 查找 CUDA 工具包
# =======================================================
if(GGML_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# 定义一个宏，用于安全地链接可能存在的 ggml 子目标
# 新版 llama.cpp 拆分了很多库，我们需要把它们都链接进来
macro(safe_link_ggml target_name)
    if(TARGET ggml)
        target_link_libraries(${target_name} PRIVATE ggml)
    endif()
    if(TARGET ggml-base)
        target_link_libraries(${target_name} PRIVATE ggml-base)
    endif()
    if(TARGET ggml-cuda) # 这是 CUDA 后端的核心
        target_link_libraries(${target_name} PRIVATE ggml-cuda)
    endif()
    if(TARGET ggml-cpu)
        target_link_libraries(${target_name} PRIVATE ggml-cpu)
    endif()
endmacro()

# -------------------------------------------------------
# 1) diffusion_sampler 静态库
# -------------------------------------------------------
set(DIFFUSION_SAMPLER_SRCS
    diffusion_sampler.cpp
    diffusion_sampler.h
    diffusion_types.h
)

if(GGML_CUDA)
    list(APPEND DIFFUSION_SAMPLER_SRCS
        gpu_sampler.cu
        gpu_sampler.h
    )
endif()

add_library(diffusion_sampler STATIC
    ${DIFFUSION_SAMPLER_SRCS}
)
set_target_properties(diffusion_sampler PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(diffusion_sampler PUBLIC
    ${LLAMA_INCLUDE}
    ${LLAMA_SRC}
    ${LLAMA_ROOT}/include
    ${LLAMA_COMMON}
)
target_link_libraries(diffusion_sampler PUBLIC llama) # 已经是关键字 PUBLIC
safe_link_ggml(diffusion_sampler)
if(GGML_CUDA)
    target_compile_definitions(diffusion_sampler PUBLIC DIFFUSION_ENABLE_CUDA=1)
endif()

# -------------------------------------------------------
# 2) Profiled version (性能分析版)
# -------------------------------------------------------
set(DIFFUSION_SAMPLER_PROFILED_SRCS
    diffusion_sampler_profiled.cpp
    diffusion_sampler.cpp
)

if(GGML_CUDA)
    list(APPEND DIFFUSION_SAMPLER_PROFILED_SRCS
        gpu_sampler.cu
        gpu_sampler.h
    )
endif()

add_library(diffusion_sampler_profiled STATIC
    ${DIFFUSION_SAMPLER_PROFILED_SRCS}
)
set_target_properties(diffusion_sampler_profiled PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(diffusion_sampler_profiled PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LLAMA_INCLUDE}
    ${LLAMA_SRC}
    ${LLAMA_ROOT}/include
    ${LLAMA_COMMON}
)
# 【这里是修改点】将 plain signature 改为 keyword signature
target_link_libraries(diffusion_sampler_profiled PRIVATE llama) 
safe_link_ggml(diffusion_sampler_profiled)
if(GGML_CUDA)
    target_compile_definitions(diffusion_sampler_profiled PUBLIC DIFFUSION_ENABLE_CUDA=1)
endif()

# -------------------------------------------------------
# 3) Python Bindings for Profiled Version
# -------------------------------------------------------
pybind11_add_module(llama_diffusion_profiled python_bindings_profiled.cpp)

set_target_properties(llama_diffusion_profiled PROPERTIES
    PREFIX ""
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(llama_diffusion_profiled PRIVATE
    diffusion_sampler_profiled
    llama
)
safe_link_ggml(llama_diffusion_profiled)

if(GGML_CUDA)
    target_link_libraries(llama_diffusion_profiled PRIVATE 
        CUDA::cudart
        CUDA::cublas
        CUDA::cublasLt
    )
endif()

# -------------------------------------------------------
# 4) Standard Version (Android / Desktop)
# -------------------------------------------------------
if(ANDROID)
    add_library(llama_diffusion SHARED jni_wrapper.cpp)
    set_target_properties(llama_diffusion PROPERTIES PREFIX "" OUTPUT_NAME "llama_diffusion")
    target_include_directories(llama_diffusion PRIVATE ${LLAMA_INCLUDE} ${LLAMA_SRC} ${LLAMA_ROOT}/include ${LLAMA_COMMON})
    target_link_libraries(llama_diffusion diffusion_sampler llama log android)
    safe_link_ggml(llama_diffusion)

else()
    # 4B) 桌面端：生成 Python 模块
    pybind11_add_module(llama_diffusion
        diffusion_binding.cpp
    )
    set_target_properties(llama_diffusion PROPERTIES
        PREFIX ""
        POSITION_INDEPENDENT_CODE ON
    )
    target_link_libraries(llama_diffusion PRIVATE
        diffusion_sampler
        llama
    )
    safe_link_ggml(llama_diffusion)
    
    if(GGML_CUDA)
        target_link_libraries(llama_diffusion PRIVATE 
            CUDA::cudart 
            CUDA::cublas 
            CUDA::cublasLt
        )
    endif()
endif()
