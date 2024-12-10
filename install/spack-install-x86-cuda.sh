#!/usr/bin/bash
#
#  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
#                      All rights reserved.
#

git clone -c feature.manyFiles=true https://github.com/spack/spack.git
. $HOME/spack/share/spack/setup-env.sh
spack compiler find
spack env create tlrmvm
spack env activate tlrmvm
spack cd -e tlrmvm
echo '# This is a Spack Environment file.
#
# It describes a set of packages to be installed, along with
# configuration settings.
spack:
  packages:
    all:
      target: [x86_64]
      compiler: [gcc@11.4.0]
    cmake:
      version: [3.21.7]
  specs:
  - cmake@3.21.7
  - intel-oneapi-mkl
  - intel-oneapi-compilers
  - openmpi
  - cuda@11.5.1~allow-unsupported-compilers~dev
  view: true
' > spack.yaml
spack concretize
spack install