include(${PROJECT_SOURCE_DIR}/cmake/tlrmvm.cmake)
set(HIP_SRCS)
set(HIP_HEADERS)

# Common
file(GLOB TMP ${PROJECT_SOURCE_DIR}/src/common/hip/*.cpp)
list(APPEND HIP_SRCS ${TMP})
file(GLOB TMP ${PROJECT_SOURCE_DIR}/src/common/hip/*.cu)
list(APPEND HIP_SRCS ${TMP})
file(GLOB TMP ${PROJECT_SOURCE_DIR}/src/common/hip/*.hpp)
list(APPEND HIP_HEADERS ${TMP})

# tlrmvm
file(GLOB TMP ${PROJECT_SOURCE_DIR}/src/tlrmvm/hip/*.cpp)
list(APPEND HIP_SRCS ${TMP})
file(GLOB TMP ${PROJECT_SOURCE_DIR}/src/tlrmvm/hip/*.cu)
list(APPEND HIP_SRCS ${TMP})
file(GLOB TMP ${PROJECT_SOURCE_DIR}/src/tlrmvm/hip/*.hpp)
list(APPEND HIP_HEADERS ${TMP})
file(GLOB TMP ${PROJECT_SOURCE_DIR}/src/tlrmvm/hip/*.cuh)
list(APPEND HIP_HEADERS ${TMP})