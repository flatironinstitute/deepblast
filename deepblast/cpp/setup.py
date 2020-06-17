from setuptools import setup, Extension
from torch.utils import cpp_extension


setup(name='nw_cpp',
      ext_modules=[cpp_extension.CppExtension(
          'nw_cpp', ['nw.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
