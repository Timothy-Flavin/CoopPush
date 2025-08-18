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

        # --- Signal that this is NOT a pure Python wheel ---
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

        # Build into the 'src' directory, and setuptools will create 'src/cooppush'
        build_path = Path(self.root) / "src"
        cmd.build_lib = str(build_path.resolve())

        print("--- Compiling C++ extension ---")
        cmd.run()
        print("--- Compilation finished ---")

        # --- NEW: Use force_include for a more robust approach ---
        # This explicitly maps the compiled file to its destination in the wheel.
        build_data["force_include"] = {}
        artifacts_path = build_path / "cooppush"

        for p in artifacts_path.glob("cooppush_cpp*"):
            # The key is the full path to the source file we just built.
            source = str(p)
            # The value is the destination path inside the wheel's 'cooppush' package.
            destination = str(Path("cooppush") / p.name)
            print(f"--- Mapping artifact: {source} -> {destination} ---")
            build_data["force_include"][source] = destination
