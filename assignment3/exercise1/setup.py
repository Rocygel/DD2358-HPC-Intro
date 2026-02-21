from distutils.core import setup
from Cython.Build import cythonize
import numpy
import torch
import pytorch_ocl

setup(
    ext_modules=cythonize(
        "./*.pyx", 
        compiler_directives={"language_level": "3"}
    ),
    include_dirs=[numpy.get_include()]
)
