import os
import sys
from pathlib import Path
import pybind11
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        """
        This method is called before the build process starts.
        This is where we compile the C++ extension.
        """
        from setuptools import Extension
        from setuptools.command.build_ext import build_ext

        # Define the extension module, just like in setup.py
        ext_modules = [
            Extension(
                "cooppush.cpp_backend",
                ["cpp_src/backend.cpp"],
                include_dirs=[pybind11.get_include()],
                language="c++",
                extra_compile_args=(
                    ["-std=c++11"] if sys.platform != "win32" else ["/std:c++14"]
                ),
            )
        ]

        # A small distribution object to run the build_ext command
        from distutils.dist import Distribution

        dist = Distribution({"name": "cooppush", "ext_modules": ext_modules})

        # Create a build_ext command and configure it
        cmd = build_ext(dist)
        cmd.ensure_finalized()

        # The output directory for the compiled file
        build_lib = Path(self.root) / "cooppush"
        cmd.build_lib = str(build_lib.resolve())

        print("--- Compiling C++ extension ---")
        cmd.run()
        print("--- Compilation finished ---")
