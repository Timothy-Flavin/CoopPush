# setup.py

import sys
from setuptools import setup, Extension, find_packages
import pybind11

# Define the C++ extension module
ext_modules = [
    Extension(
        # The name of the extension, including the package path
        "cooppush.cooppush_cpp",
        # A list of C++ source files
        ["cpp_src/backend.cpp"],
        # Include directories for header files
        include_dirs=[pybind11.get_include()],
        language="c++",
        # Platform-specific compiler arguments
        extra_compile_args=(
            ["-std=c++11"] if sys.platform != "win32" else ["/std:c++14"]
        ),
    ),
]

# The main setup call
setup(
    # These two lines are crucial for finding your Python source code in the 'src' dir
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    # This tells setuptools to build the C++ extension we defined above
    ext_modules=ext_modules,
)
