import os

import numpy
from setuptools import Extension, setup

llvm_dir = "/opt/homebrew/opt/llvm"  # Intel Macなら "/usr/local/opt/llvm"

dtw_module = Extension(
    "dtwmodule",
    sources=["dtw.c", "dtwmodule.c"],
    include_dirs=[numpy.get_include(), os.path.join(llvm_dir, "include")],
    library_dirs=[os.path.join(llvm_dir, "lib")],
    extra_compile_args=["-Xpreprocessor", "-fopenmp"],
    extra_link_args=["-lomp"],
)

setup(name="dtwmodule", version="1.0", ext_modules=[dtw_module])
