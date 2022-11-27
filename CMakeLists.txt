cmake_minimum_required(VERSION 3.20)

set(PROJECT_NAME TLRMVM)

PROJECT(${PROJECT_NAME} C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "-Wunused-result")
set(CMAKE_C_FLAGS "-Wunused-result")
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
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)


###################
# Build options
###################
# choose the platform you want to deploy TLR-MVM, only one of below 4 choices should be set as ON.
option(BUILD_CPU "Build TLR-MVM using cpp" ON)
option(BUILD_DPCPP "Build TLR-MVM on intel archs and use intel dpc++." OFF)
option(BUILD_CUDA "Build TLR-MVM on NVIDIA gpu and cuda." OFF) # using NVIDIA GPU
 option(BUILD_HIP "Build TLR-MVM on amd gpu and use hip." OFF) # using AMD GPU (AMD is under dev)

#########################
# BLAS backend selection
#########################
# TLR-MVM uses sequential blas implementation, but if we want to benchmark with mvm,
# we should also find a threaded version blas implementation. Only one of them can be
# activated.
option(USE_MKL "USE MKL" OFF) # any platform suitable for MKL
option(USE_BLIS "USE AMD BLIS" OFF) # usually for amd epyc cpu.
option(USE_OPENBLAS "USE OpenBLAS" OFF) # any platform that can install openblas
option(USE_COMPILER_BLAS "Use compiler wrapper blas implementation, 
on Shaheen/Fujitsu/NEC etc." OFF) # for nec / fujitsu

# python support.
option(BUILD_PYTHON "Build Python interface for TLR-MVM, 
we recommend to use setup.py for python install." OFF)

# Test TLR-MVM correctness.
option(BUILD_TEST "Build Test." OFF)

# Build documentation, set OFF, since the documentation is under development.
option(BUILD_DOC "Build Documentation" OFF)

# MPI interface, find openmpi library, if we are on shaheen/fujitsu/nec, we should
# omit this step.
option(USE_MPI "Build with MPI." OFF)

# OpenMP support.
option(USE_OpenMP "Build with OpenMP." ON)


set(TLRMVM_INC "")
set(TLRMVM_LIBS "")


##################
# BLAS Backend
##################
if(USE_MKL)
    message(STATUS "Using MKL: ")
    message($ENV{MKLROOT})
    set(BLAS_INC $ENV{MKLROOT}/include)
    # sequential mkl
    set(BLA_VENDOR Intel10_64lp_seq)
    find_package(BLAS REQUIRED)
    set(TLRMVM_INCS ${TLRMVM_INCS} $ENV{MKLROOT}/include)
    set(TLRMVM_LIBS ${TLRMVM_LIBS} ${BLAS_LIBRARIES})
    message(${BLAS_LIBRARIES})
    set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} $ENV{MKLROOT}/lib $ENV{MKLROOT}/lib/intel64)
endif() # MKL

if(USE_BLIS)
    message(STATUS "Using BLIS Library: ")
    message($ENV{BLIS_ROOT})
    list(APPEND TLRMVM_INCS $ENV{BLIS_ROOT}/include/blis)
    LIST(APPEND TLRMVM_LIBS $ENV{BLIS_ROOT}/lib/libblis.so)
    set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} $ENV{BLIS_ROOT}/lib)
endif() # BLIS

if(USE_OPENBLAS)
    message(STATUS "Using OpenBLAS Library: ")
    message({$ENV{OPENBLAS_ROOT})
    set(APPEND TLRMVM_INCS $ENV{OPENBLAS_ROOT}/include)
    LIST(APPEND TLRMVM_LIBS $ENV{OPENBLAS_ROOT}/lib/libopenblas.so)
    set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} $ENV{OPENBLAS_ROOT}/lib)
endif() # OPENBLAS

if (USE_COMPILER_BLAS)
    message(STATUS "Using Compiler BLAS Library: Compiler id is " ${CMAKE_C_COMPILER_ID})
    add_compile_definitions(USE_COMPILER_BLAS)
    if("${CMAKE_C_COMPILER_ID}" STREQUAL "NEC")
	add_compile_definitions(USE_NEC)
        set(BLAS_LINK_OPT "-lcblas -lblas_sequential")
    endif() # NEC
    if("${CMAKE_C_COMPILER_ID}" STREQUAL "Fujitsu")
        add_compile_definitions(USE_FUJITSU)
        set(BLAS_COMPILE_OPT -Nclang -O3 -fPIC -KA64FX,ARMV8_3_A,openmp)
        set(BLAS_LINK_OPT -SSL2 -lstdc++)
    endif() # Fujitu
endif() # USE_COMPILER_BLAS


#################
# OpenMP
#################
find_package(OpenMP)
# set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
if(BUILD_DPCPP)
    # disable openmp
else()
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


#################
# MPI Backend
#################
if(USE_MPI)
    if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_C_COMPILER_ID}" STREQUAL "INTEL")
        set (MPI_HOME "$ENV{MPI_ROOT}")
        find_package(MPI REQUIRED)
        set(TLRMVM_INCS ${TLRMVM_INCS} ${MPI_C_INCLUDE_PATH})
        set(TLRMVM_LIBS ${TLRMVM_LIBS} ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
        set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} $ENV{MPI_ROOT}/lib)
    elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "NEC")
        set(TLRMVM_LIBS ${TLRMVM_LIBS} -lmpi -lmpi++)
    endif()
endif() # USE MPI
ShowCompilerinfo()

######################
# CPU library
######################
if(BUILD_CPU)
    include(cmake/tlrmvm.cmake)
    add_library(tlrmvmcpulib SHARED ${CPU_HEADERS} ${CPU_SRCS})
    target_include_directories(tlrmvmcpulib PUBLIC ${TLRMVM_INCS})
    target_link_libraries(tlrmvmcpulib PUBLIC ${TLRMVM_LIBS})
    target_compile_options(tlrmvmcpulib PUBLIC ${BLAS_COMPILE_OPT})
    target_link_options(tlrmvmcpulib PUBLIC ${BLAS_LINK_OPT})
    AddCompileDefinitions(tlrmvmcpulib)
    install(TARGETS tlrmvmcpulib DESTINATION lib)
endif()


#######################
# CUDA library
#######################
if(BUILD_CUDA)
    find_package(CUDAToolkit 11.0 REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES 60 70 80)
    set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
    enable_language(CUDA)
    # add cuda lib dep
    set(TLRMVM_INCS ${TLRMVM_INCS} ${CUDAToolkit_INCLUDE_DIRS})
    set(TLRMVM_LIBS ${TLRMVM_LIBS} CUDA::cublas CUDA::cudart)
    # build cuda library
    include(cmake/cudatlrmvm.cmake)
    set_source_files_properties(${CUDA_SRCS} PROPERTIES LANGUAGE CUDA)
    add_library(tlrmvmcudalib SHARED ${CPU_HEADERS} ${CPU_SRCS} ${CUDA_HEADERS} ${CUDA_SRCS})
    target_include_directories(tlrmvmcudalib PUBLIC ${TLRMVM_INCS})
    target_link_libraries(tlrmvmcudalib PUBLIC ${TLRMVM_LIBS})
    AddCompileDefinitions(tlrmvmcudalib)
    target_compile_definitions(tlrmvmcudalib PUBLIC USE_CUDA)
    install(TARGETS tlrmvmcudalib DESTINATION lib)
endif() # BUILD_CUDA


#######################
# HIP library
#######################
if(BUILD_HIP)
    enable_language(HIP)
    find_package(HIP REQUIRED)
    include(cmake/hiptlrmvm.cmake)
    add_library(tlrmvmhiplib SHARED ${CPU_HEADERS} ${CPU_SRCS} ${HIP_HEADERS} ${HIP_SRCS})
#    find_package(hipblas REQUIRED)
    set(TLRMVM_LIBS ${TLRMVM_LIBS} hip::device hip::host hipblas)
    target_include_directories(tlrmvmhiplib PUBLIC ${TLRMVM_INCS})
    target_link_libraries(tlrmvmhiplib PUBLIC ${TLRMVM_LIBS})
    AddCompileDefinitions(tlrmvmhiplib)
#    target_compile_definitions(tlrmvmhiplib PUBLIC -D__HIP_PLATFORM_HCC__=1) # for clion search ...
    target_compile_definitions(tlrmvmhiplib PUBLIC USE_HIP)
    install(TARGETS tlrmvmhiplib DESTINATION lib)
endif() # BUILD_HIP


#################
# Python Build
#################
if(BUILD_PYTHON AND (BUILD_CUDA OR BUILD_HIP))
    # now python is only available for CUDA and HIP build.
    find_package(pybind11 REQUIRED)
    add_subdirectory(pytlrmvm)
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
