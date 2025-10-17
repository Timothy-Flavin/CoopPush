import sys
from pathlib import Path
from setuptools import setup, Extension, find_packages
import sys
from pathlib import Path
from setuptools import setup, Extension
import pybind11

# Build a POSIX-style, relative source list to avoid absolute path errors on Windows
root = Path(__file__).resolve().parent
cpp_sources = [
    p.relative_to(root).as_posix() for p in sorted((root / "cpp_src").glob("*.cpp"))
]
header_files = [
    p.relative_to(root).as_posix() for p in sorted((root / "cpp_include").glob("*.h"))
]

ext_modules = [
    Extension(
        "cooppush.cooppush_cpp",
        cpp_sources,
        include_dirs=[
            pybind11.get_include(),
            str((Path(__file__).resolve().parent / "cpp_include")),
        ],
        depends=header_files,
        language="c++",
        extra_compile_args=(
            ["-std=c++17"] if sys.platform != "win32" else ["/std:c++17"]
        ),
    ),
]
setup(
    name="cooppush",
    version="1.2.1",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    ext_modules=ext_modules,
)
