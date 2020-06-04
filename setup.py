import glob
import setuptools
import os
import site

# See: https://stackoverflow.com/questions/54117786/add-numpy-get-include-argument-to-setuptools-without-preinstalled-numpy

# factory function
def my_build_ext(pars):
    # import delayed:
    from setuptools.command.build_ext import build_ext as _build_ext#

    # include_dirs adjusted:
    class build_ext(_build_ext):
        def finalize_options(self):
            _build_ext.finalize_options(self)
            # Prevent numpy from thinking it is still in its setup process:
            __builtins__.__NUMPY_SETUP__ = False
            import numpy
            self.include_dirs.append(numpy.get_include())

    #object returned:
    return build_ext(pars)

FLAGS = ['-JSON_DLL_BUILD', '-DSTREAMDM_EXPORTS', '-fPIC', '-shared', '-std=c++11',
         '-D_GNU_SOURCE', '-D_FILE_OFFSET_BITS=64', '-D_LARGEFILE_SOURCE64', '-O3', '-DUNIX', '-lpython']

cpp_sources = glob.glob('./code/src/**/*.cpp', recursive=True)


swig_lib = setuptools.Extension(
    name='_streamdm',
    sources=[os.path.join('code', 'src', 'streamdm_wrap.cxx'),
             *cpp_sources,
             ],
    extra_compile_args=FLAGS,
)

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='ml-rapids',  # Replace with your own username
    version='0.0.1.4',
    author='Klemen Kenda',
    author_email='klemen.kenda@ijs.si',
    description='Incremental learning written in C++ exposed in Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='GPLv3',
    url='https://github.com/JozefStefanInstitute/ml-rapids',
    packages=setuptools.find_packages(),
    setup_requires=['numpy', ],
    install_requires=['scikit-learn', ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    ext_modules=[swig_lib, ],
    extra_compile_args=FLAGS,
    python_requires='>=3.6',
    cmdclass={'build_ext' : my_build_ext},
)
