# setup.py

import sys
from setuptools import setup, Extension, find_packages
import pybind11

# Define the C++ extension module
ext_modules = [
    Extension(
        "cooppush.cooppush_cpp",
        ["cpp_src/backend.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=(
            ["-std=c++11"] if sys.platform != "win32" else ["/std:c++14"]
        ),
    ),
]
