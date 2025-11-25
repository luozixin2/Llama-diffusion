import os, sys, subprocess, shutil
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir="."):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir) # sourcedir will now be the project root

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        # 确定 CMake 构建目录
        # 为每个扩展创建一个独立的构建目录，基于 setuptools 的临时目录
        # 例如：/tmp/tmpxxx.build-temp/llama_diffusion.llama_diffusion
        build_dir = Path(self.build_temp) / ext.name 
        build_dir.mkdir(parents=True, exist_ok=True)
        
        # 确定最终的安装目录，setuptools 会将 .so 文件复制到这里
        # 例如：/tmp/tmpyyy.build-lib/llama_diffusion/llama_diffusion/
        ext_full_dest_path = Path(self.get_ext_fullpath(ext.name))
        dst_path = ext_full_dest_path
        
        # 1) 配置 CMake
        # 【关键修改 1】让 CMake 从项目的根目录开始配置
        cmake_sourcedir = os.path.abspath(os.path.dirname(__file__)) # 项目根目录
        
        cmake_args = [
            "cmake", cmake_sourcedir, # 指向顶层 CMakeLists.txt
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DBUILD_SHARED_LIBS=OFF", # 我们编译的是静态库，但pybind11模块是共享库
            f"-DCMAKE_BUILD_TYPE={self.build_type}", # 使用 setuptools 的 build_type (Debug/Release)
            "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
            # 【可选】如果你想让 CMake 将所有输出都放到 build_dir 的根目录，可以加上这个
            # 但通常 pybind11_add_module 会处理好输出路径
            # f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={build_dir}", 
        ]

        # 如果是 Windows，可能需要指定生成器
        if sys.platform == "win32":
            cmake_args += ["-G", "Visual Studio 16 2019"] # 或其他版本

        print(f"Configuring CMake in {build_dir} with args: {' '.join(cmake_args)}")
        subprocess.check_call(cmake_args, cwd=build_dir)
        
        # 2) 构建 CMake 项目
        # 【关键修改 2】指定要构建的特定目标
        build_args = ["cmake", "--build", ".", "--config", self.build_type]
        
        # 根据扩展名，选择对应的 CMake 目标
        # 这些目标名称来自你的 llama_diffusion/CMakeLists.txt 中的 pybind11_add_module
        cmake_target_name = None
        if ext.name == "llama_diffusion.llama_diffusion":
            cmake_target_name = "llama_diffusion"
        elif ext.name == "llama_diffusion.llama_diffusion_profiled":
            cmake_target_name = "llama_diffusion_profiled"
        else:
            raise ValueError(f"Unknown extension name: {ext.name}")
        
        build_args += ["--target", cmake_target_name]

        if not sys.platform.startswith("win"):
            build_args += ["--", f"-j{os.cpu_count()}"] # 并行编译
        
        print(f"Building CMake project in {build_dir} with args: {' '.join(build_args)}")
        subprocess.check_call(build_args, cwd=build_dir)
        
        # 3) 【核心修改】手动复制编译好的 .so 文件到 setuptools 期望的位置
        # 获取最终的 .so 文件名部分 (例如：llama_diffusion.cpython-311-x86_64-linux-gnu.so)
        ext_base_filename = ext_full_dest_path.name 

        # 当顶层 CMakeLists.txt 包含 add_subdirectory(llama_diffusion) 时，
        # 针对 llama_diffusion 子目录中定义的 target (如 llama_diffusion 或 llama_diffusion_profiled)
        # 它们的构建产物通常会放在 build_dir/<subdirectory_name>/ 目录下。
        # 在本例中，<subdirectory_name> 就是 "llama_diffusion"。
        cmake_subproject_build_dir_name = "llama_diffusion" 

        # 尝试查找输出文件可能的位置
        possible_src_paths = [
            # 最常见的位置： build_dir/<subproject_name>/<filename>
            build_dir / cmake_subproject_build_dir_name / ext_base_filename,
            # 对于多配置生成器 (如 Windows 上的 Visual Studio)，可能在 config 子目录下
            build_dir / cmake_subproject_build_dir_name / self.build_type / ext_base_filename,
            # 有时也可能直接在 build_dir 根目录下 (如果 CMake 的输出目录被重定向)
            build_dir / ext_base_filename, 
            # 如果 target name 和 subproject dir name 相同，并且没有 add_subdirectory
            # 但我们这里有 add_subdirectory，所以这个可能性较低
            build_dir / cmake_target_name / ext_base_filename,
            build_dir / cmake_target_name / self.build_type / ext_base_filename,
        ]

        src_path = None
        for p in possible_src_paths:
            if p.exists():
                src_path = p
                break
        
        if src_path:
            # 确保目标目录存在
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_path, dst_path)
            print(f"Successfully copied {src_path} to {dst_path}")
        else:
            # 如果仍然找不到，打印 build_dir 的内容以帮助调试
            print(f"\nERROR: Could not find the built shared library '{ext_base_filename}'.")
            print(f"Expected destination: {dst_path}")
            print(f"Searched in: {[str(p) for p in possible_src_paths]}")
            print(f"\n--- Contents of CMake build directory ({build_dir}) for debugging ---")
            if build_dir.exists():
                for root, dirs, files in os.walk(build_dir):
                    for name in files:
                        print(Path(root) / name)
            else:
                print(f"CMake build directory {build_dir} does not exist.")
            print("---------------------------------------------------------------------\n")
            raise FileNotFoundError(
                f"CMake did not produce the expected shared library '{ext_base_filename}'. "
                f"Please check the full output above for the actual output path of CMake."
            )

    # setuptools 的 build_type 属性
    def initialize_options(self):
        super().initialize_options()
        self.build_type = "Release" # 默认 Release
        if self.debug:
            self.build_type = "Debug"

setup(
    name="llama_diffusion",
    version="0.1.0",
    # 确保 Python 包目录结构正确
    packages=["llama_diffusion"], 
    ext_modules=[
        # sourcedir 应该指向项目的根目录，因为 CMakeBuild 现在从根目录开始配置
        CMakeExtension("llama_diffusion.llama_diffusion", sourcedir="."),
        CMakeExtension("llama_diffusion.llama_diffusion_profiled", sourcedir="."),
    ],
    cmdclass={"build_ext": CMakeBuild},
    package_data={
        # 包含生成的扩展文件
        "llama_diffusion": ["*.pyd", "*.so", "*.dll"],
    },
    zip_safe=False,
    python_requires=">=3.7",
)
