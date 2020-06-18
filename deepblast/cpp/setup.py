from setuptools import setup, Extension
from torch.utils import cpp_extension


setup(name='nw',
      ext_modules=[cpp_extension.CppExtension(
          'nw', ['nw.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
