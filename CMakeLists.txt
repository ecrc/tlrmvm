cmake_minimum_required(VERSION 3.18)
set(PROJECT_NAME TLRMVM)
PROJECT(${PROJECT_NAME} C CXX)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
# directly make an error if in-source build
if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
message(FATAL_ERROR "In-source builds are not allowed.\n")
endif()
# load utility functions
include(cmake/Util.cmake)
# Set the RPATH config
# --------------------
# use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
# when building, use the install RPATH already
# (automated test will need this)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
# the RPATH to be used when installing
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

###################
# Build options
###################
# Build library using different architectures
option(BUILD_DPCPP "Build library using Intel DPC++" OFF)
option(BUILD_CUDA "Build library using NVIDIA GPU" OFF)
option(BUILD_AMDGPU "Build library using AMD GPU" OFF)
option(BUILD_PYTHON "Build Python interface of library" OFF)
option(BUILD_TEST "Build Test" OFF)
option(BUILD_DOC "Build Documentation" OFF)

# BLAS backend selection
option(USE_OPENBLAS "USE OpenBLAS Library" OFF)
option(USE_MKL "USE INTEL MKL" OFF)
option(USE_COMPILER_BLAS 
"Use compiler wrapper blas impl, on shaheen/Fujitsu etc." OFF)
option(USE_BLIS "USE BLIS Library" OFF)
option(USE_MPI "Build with MPI." OFF)
option(USE_OpenMP "Build with OpenMP." ON)

# cuda profiling selection
option(USE_NVTX "USE_NVTX" OFF)
# NCCL
option(USE_NCCL "Use NCCL" OFF)

AddCompiledefinitions()

####################
# Find Dependencies
####################
message("====================TLRMVM Dependencies INFO====================")
#################
# OpenMP
#################
find_package(OpenMP)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
if(BUILD_DPCPP)
else()
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
##################
# BLAS Backend
##################
if (NOT USE_COMPILER_BLAS)
if(USE_MKL)
message(STATUS "Using MKL: ")
# sequential mkl
set(BLA_VENDOR Intel10_64lp_seq)
find_package(BLAS REQUIRED)
add_link_options(${BLAS_LINKER_FLAGS})
set(TLRMVM_LIBS ${TLRMVM_LIBS} ${BLAS_LIBRARIES})
message(${BLAS_LINKER_FLAGS} ${BLAS_LIBRARIES})
if(BUILD_DPCPP)
find_library(MKLSYCL NAMES mkl_sycl HINTS "$ENV{MKLROOT}/lib/intel64" REQUIRED)
set(TLRMVM_LIBS ${TLRMVM_LIBS} ${MKLSYCL})
endif()
endif() # MKL
if(USE_BLIS)
message(STATUS "Using BLIS Library: ")
message($ENV{BLIS_ROOT})
set(TLRMVM_INCS ${TLRMVM_INCS} $ENV{BLIS_ROOT}/include/blis)
set(TLRMVM_LIBS ${TLRMVM_LIBS} $ENV{BLIS_ROOT}/lib/libblis.so)
endif() # BLIS
if(USE_OPENBLAS)
message(STATUS "Using OpenBLAS Library: ")
message($ENV{OPENBLAS_ROOT})
set(TLRMVM_INCS ${TLRMVM_INCS} $ENV{OPENBLAS_ROOT}/include)
set(TLRMVM_LIBS ${TLRMVM_LIBS} $ENV{OPENBLAS_ROOT}/lib/libopenblas.so)
endif() # OPENBLAS
else()
message(STATUS "Using Compiler BLAS Library: ")
add_compile_definitions(USE_COMPILER_BLAS)
if("${CMAKE_C_COMPILER_ID}" STREQUAL "NEC")
add_link_options(-lcblas -lblas_sequential)
endif() # NEC
if("${CMAKE_C_COMPILER_ID}" STREQUAL "Fujitsu")
add_compile_options(-KA64FX -O3 -KSVE -KARMV8_3_A -Kopenmp -fPIC)
add_link_options(-SSL2)
endif() # Fujitu
if("${CMAKE_C_COMPILER_ID}" STREQUAL "AppleClang")
add_compile_definitions(BUILD_OSX)
set(BLA_VENDOR Apple)
find_package(BLAS REQUIRED)
add_link_options(${BLAS_LINKER_FLAGS})
set(TLRMVM_INCS ${TLRMVM_INCS} ${BLAS_LIBRARIES}/Headers)
set(TLRMVM_LIBS ${TLRMVM_LIBS} ${BLAS_LIBRARIES} -lstdc++)
endif() # Apple

endif() # USE_COMPILER_BLAS


#################
# MPI Backend
#################
if(USE_MPI)
if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_C_COMPILER_ID}" STREQUAL "INTEL")
set (MPI_HOME "$ENV{MPI_ROOT}")
find_package(MPI REQUIRED)
set(TLRMVM_INCS ${TLRMVM_INCS} ${MPI_C_INCLUDE_PATH})
set(TLRMVM_LIBS ${TLRMVM_LIBS} ${MPI_C_LIBRARIES})
elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "NEC")
set(TLRMVM_LIBS ${TLRMVM_LIBS} -lmpi -lmpi++)
endif()
endif() # USE MPI
ShowCompilerinfo()
message("====================TLRMVM Dependencies INFO END====================")



######################
# CPU library
######################
set(TLRMVM_INCS ${TLRMVM_INCS} ${CMAKE_SOURCE_DIR}/src)
add_subdirectory(src/common)
add_subdirectory(src/tlrmvm/cpu)
add_library(tlrmvmcpulib SHARED ${TLRMVM_SRCS})
target_include_directories(tlrmvmcpulib PUBLIC ${TLRMVM_INCS})
target_link_libraries(tlrmvmcpulib PUBLIC ${TLRMVM_LIBS})
install(TARGETS tlrmvmcpulib DESTINATION lib)

#######################
# CUDA library
#######################
if(BUILD_CUDA)
message("====================CUDA INFO============================")
if(USE_NCCL)
message($ENV{NCCL_ROOT})
set(TLRMVM_INCS ${TLRMVM_INCS} $ENV{NCCL_ROOT}/include)
set(TLRMVM_LIBS ${TLRMVM_LIBS} $ENV{NCCL_ROOT}/lib/libnccl.so)
endif()
set(CUDAToolkit_ROOT $ENV{CUDAToolkit_ROOT})
find_package(CUDAToolkit 11.0 REQUIRED)
enable_language(CUDA)
OutputCUDAinfo()
message("=========================================================")
set(CMAKE_CUDA_ARCHITECTURES 60 70 80)
# add cuda lib dep
set(TLRMVM_INCS ${TLRMVM_INCS} ${CUDAToolkit_INCLUDE_DIRS})
set(TLRMVM_LIBS ${TLRMVM_LIBS} ${CUDAToolkit_LIBRARY_DIR}/libcublas.so 
${CUDAToolkit_LIBRARY_DIR}/libcudart.so)

# build cuda library
set(CUDATLRMVM_SRCS)
add_subdirectory(src/common/cuda)
add_subdirectory(src/tlrmvm/cuda)
set_source_files_properties(${CUDATLRMVM_SRCS} PROPERTIES LANGUAGE CUDA)
add_library(tlrmvmcudalib SHARED ${TLRMVM_SRCS} ${CUDATLRMVM_SRCS})
target_include_directories(tlrmvmcudalib PUBLIC ${TLRMVM_INCS})
target_link_libraries(tlrmvmcudalib PUBLIC ${TLRMVM_LIBS})
install(TARGETS tlrmvmcudalib DESTINATION lib)
endif() # BUILD_CUDA

#################
# Python Build
#################
if(BUILD_PYTHON)
add_subdirectory(thirdparty/pybind11)
add_subdirectory(python)
endif()

#################
# Test Build
#################
if(BUILD_TEST)
add_subdirectory(test)
endif()

######################
# Documentation Build
######################
if(BUILD_DOC)
add_subdirectory(doxygen)
endif()
