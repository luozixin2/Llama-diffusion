#pragma once
#include <cstdio>

// 日志等级：F(致命)、E(错误)、W(警告)、I(信息)、D(调试)、V(啰嗦)
#define DIFF_LOG_LEVEL_FATAL   0
#define DIFF_LOG_LEVEL_ERROR   1
#define DIFF_LOG_LEVEL_WARN    2
#define DIFF_LOG_LEVEL_INFO    3
#define DIFF_LOG_LEVEL_DEBUG   4
#define DIFF_LOG_LEVEL_VERBOSE 5

// 编译期配置：默认 INFO（可通过 -DDIFFUSION_LOG_LEVEL=DIFF_LOG_LEVEL_xxx 覆盖）
#ifndef DIFFUSION_LOG_LEVEL
#define DIFFUSION_LOG_LEVEL DIFF_LOG_LEVEL_INFO
#endif

// 核心日志宏：在编译期比较等级，低于阈值的直接编译掉，避免格式化开销
#define DIFF_LOG_AT(level, fmt, ...)                                                   \
    do {                                                                               \
        if ((level) <= DIFFUSION_LOG_LEVEL) {                                          \
            std::fprintf(stderr, fmt, ##__VA_ARGS__);                                  \
            if ((level) <= DIFF_LOG_LEVEL_WARN) std::fflush(stderr);                   \
        }                                                                              \
    } while (0)

#define DIFF_LOGF(fmt, ...) DIFF_LOG_AT(DIFF_LOG_LEVEL_FATAL,   fmt, ##__VA_ARGS__)
#define DIFF_LOGE(fmt, ...) DIFF_LOG_AT(DIFF_LOG_LEVEL_ERROR,   fmt, ##__VA_ARGS__)
#define DIFF_LOGW(fmt, ...) DIFF_LOG_AT(DIFF_LOG_LEVEL_WARN,    fmt, ##__VA_ARGS__)
#define DIFF_LOGI(fmt, ...) DIFF_LOG_AT(DIFF_LOG_LEVEL_INFO,    fmt, ##__VA_ARGS__)
#define DIFF_LOGD(fmt, ...) DIFF_LOG_AT(DIFF_LOG_LEVEL_DEBUG,   fmt, ##__VA_ARGS__)
#define DIFF_LOGV(fmt, ...) DIFF_LOG_AT(DIFF_LOG_LEVEL_VERBOSE, fmt, ##__VA_ARGS__)

