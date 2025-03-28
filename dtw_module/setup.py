import numpy
from setuptools import Extension, setup

ext_modules = [
    Extension(
        "fastdtwmodule",
        sources=["fastdtwmodule.c"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

setup(name="fastdtwmodule", version="0.1", ext_modules=ext_modules)
