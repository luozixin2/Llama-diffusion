#!/bin/bash
# SDAR Llama-Diffusion Android Build Script
# 
# Prerequisites:
#   - Android NDK installed (r25+ recommended)
#   - Set ANDROID_NDK environment variable
#
# Usage:
#   ./build_android.sh [arm64-v8a|armeabi-v7a|x86_64] [Release|Debug]
#
# Example:
#   ANDROID_NDK=/path/to/ndk ./build_android.sh arm64-v8a Release

set -e

# Default values
ABI="${1:-arm64-v8a}"
BUILD_TYPE="${2:-Release}"
API_LEVEL="${ANDROID_API_LEVEL:-24}"

# Validate NDK path
if [ -z "$ANDROID_NDK" ]; then
    echo "Error: ANDROID_NDK environment variable not set"
    echo "Please set it to your Android NDK installation path"
    echo "Example: export ANDROID_NDK=/path/to/android-ndk-r25c"
    exit 1
fi

if [ ! -d "$ANDROID_NDK" ]; then
    echo "Error: Android NDK not found at $ANDROID_NDK"
    exit 1
fi

echo "============================================"
echo "SDAR Llama-Diffusion Android Build"
echo "============================================"
echo "NDK:        $ANDROID_NDK"
echo "ABI:        $ABI"
echo "Build Type: $BUILD_TYPE"
echo "API Level:  $API_LEVEL"
echo "============================================"

# Create build directory
BUILD_DIR="build-android-${ABI}"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Find CMake toolchain file
TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake"
if [ ! -f "$TOOLCHAIN_FILE" ]; then
    echo "Error: CMake toolchain file not found at $TOOLCHAIN_FILE"
    exit 1
fi

# Configure with CMake
echo ""
echo "Configuring..."
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \
    -DANDROID_ABI="$ABI" \
    -DANDROID_PLATFORM="android-$API_LEVEL" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DANDROID_STL=c++_shared \
    -DGGML_CUDA=OFF \
    -DBUILD_PYTHON_BINDINGS=OFF

# Build
echo ""
echo "Building..."
cmake --build . --config "$BUILD_TYPE" -j$(nproc)

echo ""
echo "============================================"
echo "Build completed successfully!"
echo "============================================"
echo ""
echo "Output files:"
find . -name "*.so" -o -name "*.a" | head -20

echo ""
echo "To use in your Android project:"
echo "  1. Copy the .so files to app/src/main/jniLibs/$ABI/"
echo "  2. Include headers from llama_diffusion/"
echo "  3. Link against the library in your CMakeLists.txt"
echo ""

