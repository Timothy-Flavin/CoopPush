# hatch_build.py
import sys
from pathlib import Path
import pybind11
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    # The 'finalize' hook runs at the correct time: right before the wheel is built.
    def finalize(self, version, build_data, artifact_path):
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

        build_path = Path(self.root) / "src"
        cmd.build_lib = str(build_path.resolve())

        print("--- [finalize] Compiling C++ extension ---")
        cmd.run()
        print("--- [finalize] Compilation finished ---")

        # --- Use force_include to explicitly map the compiled file ---
        build_data["force_include"] = {}
        artifacts_path = build_path / "cooppush"

        for p in artifacts_path.glob("cooppush_cpp*"):
            source = str(p)
            destination = str(Path("cooppush") / p.name)
            print(f"--- [finalize] Mapping artifact: {source} -> {destination} ---")
            build_data["force_include"][source] = destination
