# hatch_build.py
import sys
from pathlib import Path
import pybind11
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        from setuptools import Extension
        from setuptools.command.build_ext import build_ext as _build_ext
        from distutils.dist import Distribution

        print("--- [Initialize] Entering custom build hook ---")

        ext_modules = [
            Extension(
                "cooppush.cooppush_cpp",
                ["cpp_src/backend.cpp"],
                include_dirs=[pybind11.get_include()],
                language="c++",
                extra_compile_args=(
                    ["-std=c++11"] if sys.platform != "win32" else ["/std:c++14"]
                ),
            )
        ]

        dist = Distribution({"name": "cooppush", "ext_modules": ext_modules})
        cmd = _build_ext(dist)
        cmd.ensure_finalized()

        build_path = Path(self.root) / "src"
        cmd.build_lib = str(build_path.resolve())

        print("--- [Initialize] Compiling C++ extension ---")
        cmd.run()
        print("--- [Initialize] Compilation finished ---")
