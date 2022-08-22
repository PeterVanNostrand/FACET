from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np

extra_compile_args = ['-O3', '-ffast-math', '-march=native', '-fopenmp']

exts = [
    Extension('fastcrf.rfocse_main', ['fastcrf/rfocse_main.pyx'], extra_compile_args=extra_compile_args),
    Extension('fastcrf.extraction_problem', ['fastcrf/extraction_problem.pyx'], extra_compile_args=extra_compile_args),
    Extension('fastcrf.extractor', ['fastcrf/extractor.pyx'],
              extra_compile_args=extra_compile_args),
    Extension('fastcrf.observations', ['fastcrf/observations.pyx'], extra_compile_args=extra_compile_args),
    Extension('fastcrf.splitter', ['fastcrf/splitter.pyx'], extra_compile_args=extra_compile_args),
    Extension('fastcrf.math_utils', ['fastcrf/math_utils.pyx'],
              extra_compile_args=extra_compile_args, include_dirs=[np.get_include()]),
    Extension('fastcrf.debug', ['fastcrf/debug.pyx'], extra_compile_args=extra_compile_args),
    Extension('fastcrf.debug_splitter', ['fastcrf/debug_splitter.pyx'], extra_compile_args=extra_compile_args),
]

compiler_directives = {
    'boundscheck': False,
    'wraparound': False,
    'nonecheck': False,
    'cdivision': True,
    'profile': True
}

setup(
    name='fastcrf',
    version="0.1alpha",
    ext_modules=cythonize(exts, annotate=True, compiler_directives=compiler_directives),
    include_dirs=[np.get_include()],
    packages=['fastcrf'],
    install_requires=['numpy>=1.11.0', 'scikit-learn>=0.21.0', 'ttictoc>=0.4.1', 'Cython>=0.29.14'],
    author="rrunix",
    author_email="ruben.rrf93@gmail.com",
    license="BSD",
    description="Extract counterfactual from Random Forest using model internals",
    keywords="Counterfactual Random_Forest"
)
