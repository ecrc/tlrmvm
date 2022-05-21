include(${PROJECT_SOURCE_DIR}/cmake/tlrmvm.cmake)

set(CUDA_SRCS)
set(CUDA_HEADERS)

# Common
file(GLOB TMP ${PROJECT_SOURCE_DIR}/src/common/cuda/*.cpp)
list(APPEND CUDA_SRCS ${TMP})
file(GLOB TMP ${PROJECT_SOURCE_DIR}/src/common/cuda/*.cu)
list(APPEND CUDA_SRCS ${TMP})
file(GLOB TMP ${PROJECT_SOURCE_DIR}/src/common/cuda/*.hpp)
list(APPEND CUDA_HEADERS ${TMP})

# tlrmvm
file(GLOB TMP ${PROJECT_SOURCE_DIR}/src/tlrmvm/cuda/*.cpp)
list(APPEND CUDA_SRCS ${TMP})
file(GLOB TMP ${PROJECT_SOURCE_DIR}/src/tlrmvm/cuda/*.cu)
list(APPEND CUDA_SRCS ${TMP})
file(GLOB TMP ${PROJECT_SOURCE_DIR}/src/tlrmvm/cuda/*.hpp)
list(APPEND CUDA_HEADERS ${TMP})
file(GLOB TMP ${PROJECT_SOURCE_DIR}/src/tlrmvm/cuda/*.cuh)
list(APPEND CUDA_HEADERS ${TMP})
