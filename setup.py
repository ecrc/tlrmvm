import os
import pathlib
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as build_ext_orig


class CMakeExtension(Extension):

    def __init__(self, name):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])


class build_ext(build_ext_orig):

    def __init__(self, dist):
        super().__init__(dist)
        self.cmake_args = None
        self.nvcc_result = None
        self.python_path = str(pathlib.Path(sys.executable).parent.parent.absolute())

        def check_setting(variable, default_value):
            if variable in os.environ:
                self.variable = os.environ[variable]
            else:
                self.variable = default_value
                if self.variable is None:
                    print(variable, " set to None, Please check.")
                    exit(1)
            return self.variable

        self.build_test = check_setting(variable="BUILD_TEST", default_value="OFF")
        self.c_compiler = check_setting(variable="CC", default_value="gcc")
        self.cxx_compiler = check_setting(variable="CXX", default_value="g++")
        self.extension_dir = None
        # default Release
        self.debug = False
        if 'DEBUG' in os.environ and os.environ['DEBUG'] == '1':
            self.debug = True
        self.config = 'Debug' if self.debug else 'Release'
        if 'BUILD_TEST' in os.environ and os.environ['BUILD_TEST'] == 'ON':
            self.build_test = 'ON'
        self.python_version = '.'.join([str(sys.version_info.major), str(sys.version_info.minor)])

    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def cpp_cmake(self, ext):
        pass

    def mpi_cmake(self, ext):
        pass

    def cuda_cmake(self, ext):
        # get compiler location
        self.nvcc_result = subprocess.run(['which', 'nvcc'], stdout=subprocess.PIPE) \
            .stdout.decode('utf-8').strip('\n')
        self.cmake_args = ["-DCMAKE_CUDA_COMPILER:PATH={}".format(self.nvcc_result),
                           "-DCMAKE_INSTALL_PREFIX={}/install".format(self.build_temp.absolute()),
                           "-DCMAKE_C_COMPILER={}".format(self.c_compiler),
                           "-DCMAKE_CXX_COMPILER={}".format(self.cxx_compiler),
                           "-DCMAKE_CUDA_HOST_COMPILER={}".format(self.cxx_compiler),
                           #"-DCMAKE_CUDA_FLAGS='-ccbin {}'".format(self.cxx_compiler),
                           "-DUSE_MKL=ON",
                           "-DUSE_MPI=ON",
                           "-DBUILD_CUDA=ON",
                           "-DBUILD_TEST={}".format(self.build_test),
                           "-DBUILD_PYTHON=ON",
                           "-DPYBIND11_PYTHON_VERSION={}".format(self.python_version),
                           "-DPYTHON_EXECUTABLE={}".format(sys.executable),
                           "-DPython_ROOT_DIR={}".format(self.python_path),
                           "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + str(self.extension_dir.parent.absolute()),
                           "-DCMAKE_BUILD_TYPE={}".format(self.config)]

    def hip_cmake(self, ext):
        # get compiler location
        self.cmake_args = ["-DCMAKE_INSTALL_PREFIX={}/install".format(self.build_temp.absolute()),
                           "-DCMAKE_C_COMPILER={}".format(self.c_compiler),
                           "-DCMAKE_CXX_COMPILER={}".format(self.cxx_compiler),
                           "-DUSE_BLIS=ON",
                           "-DBUILD_HIP=ON",
                           "-DBUILD_TEST={}".format(self.build_test),
                           "-DBUILD_PYTHON=ON",
                           "-DPYBIND11_PYTHON_VERSION={}".format(self.python_version),
                           "-DPYTHON_EXECUTABLE={}".format(sys.executable),
                           "-DPython_ROOT_DIR={}".format(self.python_path),
                           "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + str(self.extension_dir.parent.absolute()),
                           "-DCMAKE_BUILD_TYPE={}".format(self.config)]

    def dpcpp_cmake(self, ext):
        pass

    def nec_cmake(self, ext):
        pass

    def fujitsu_cmake(self, ext):
        pass

    def build_cmake(self, ext):
        cwd = pathlib.Path().absolute()
        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        self.build_temp = pathlib.Path(self.build_temp)
        self.build_temp.mkdir(parents=True, exist_ok=True)
        self.extension_dir = pathlib.Path(self.get_ext_fullpath(ext.name))
        self.extension_dir.parent.mkdir(parents=True, exist_ok=True)
        if 'BUILD_CUDA' in os.environ and os.environ['BUILD_CUDA'] == 'ON':
            self.cuda_cmake(ext)
        if 'BUILD_CPU' in os.environ and os.environ['BUILD_CPU'] == 'ON':
            self.cpp_cmake(ext)
        if 'BUILD_DPCPP' in os.environ and os.environ['BUILD_DPCPP'] == 'ON':
            self.dpcpp_cmake(ext)
        if 'BUILD_HIP' in os.environ and os.environ['BUILD_HIP'] == 'ON':
            self.hip_cmake(ext)
        if 'BUILD_NEC' in os.environ and os.environ['BUILD_NEC'] == 'ON':
            self.nec_cmake(ext)
        if 'BUILD_FUJITSU' in os.environ and os.environ['BUILD_FUJITSU'] == 'ON':
            self.fujitsu_cmake(ext)
        # example of build args
        build_args = [
            '--config', self.config,
            '--',
            'VERBOSE=1',
            '-j8'
        ]

        os.chdir(str(self.build_temp))
        self.spawn(['cmake', str(cwd)] + self.cmake_args)
        self.spawn(['cmake', '--build', '.'] + build_args)
        self.spawn(['cmake', '--install', '.'])
        os.chdir(str(cwd))


if __name__ == "__main__":
    setup(
        name='pytlrmvm',
        version='0.0.1',
        packages=['pytlrmvm'],
        install_requires=[
            "numpy",
            "scipy",
            "matplotlib"
        ],
        ext_modules=[CMakeExtension('TLRMVMpy')],
        cmdclass={'build_ext': build_ext}
    )
