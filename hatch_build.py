# hatch_build.py
import os
import sys
from pathlib import Path
import pybind11
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        from setuptools import Extension
        from setuptools.command.build_ext import build_ext as _build_ext
        from distutils.dist import Distribution

        # --- Signal to Hatchling that this is NOT a pure Python wheel ---
        build_data["pure_python"] = False

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

        # --- CORRECTED LINE ---
        # Build into the 'src' directory, and setuptools will create 'src/cooppush'
        build_path = Path(self.root) / "src"
        cmd.build_lib = str(build_path.resolve())

        print("--- Compiling C++ extension ---")
        cmd.run()
        print("--- Compilation finished ---")

        # --- Inform Hatchling about the created artifacts ---
        # This allows you to remove the 'force_include' from pyproject.toml
        # It finds the compiled file inside 'src/cooppush/'
        artifacts_path = build_path / "cooppush"
        build_data["artifacts"] = [str(p) for p in artifacts_path.glob("cooppush_cpp*")]
        print(f"build artifacts: {build_data['artifacts']}")
