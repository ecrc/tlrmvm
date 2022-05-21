set(CPU_SRCS)
set(CPU_HEADERS)

## Common

file(GLOB TMP ${PROJECT_SOURCE_DIR}/src/common/cpu/*.cpp)
list(APPEND CPU_SRCS ${TMP})
file(GLOB TMP ${PROJECT_SOURCE_DIR}/src/common/cpu/vendorblas/*.cpp)
list(APPEND CPU_SRCS ${TMP})

file(GLOB TMP ${PROJECT_SOURCE_DIR}/src/common/cpu/*.hpp)
list(APPEND CPU_HEADERS ${TMP})
file(GLOB TMP ${PROJECT_SOURCE_DIR}/src/common/cpu/vendorblas/*.hpp)
list(APPEND CPU_HEADERS ${TMP})

file(GLOB TMP ${PROJECT_SOURCE_DIR}/src/common/*.cpp)
list(APPEND CPU_SRCS ${TMP})
file(GLOB TMP ${PROJECT_SOURCE_DIR}/src/common/*.hpp)
list(APPEND CPU_HEADERS ${TMP})

## tlrmvm
list(APPEND CPU_SRCS
        ${PROJECT_SOURCE_DIR}/src/tlrmvm/cpu/PCMatrix.cpp
        ${PROJECT_SOURCE_DIR}/src/tlrmvm/cpu/TlrmvmCPU.cpp
        )
list(APPEND CPU_HEADERS
        ${PROJECT_SOURCE_DIR}/src/tlrmvm/Tlrmvm.hpp
        ${PROJECT_SOURCE_DIR}/src/tlrmvm/cpu/PCMatrix.hpp
        ${PROJECT_SOURCE_DIR}/src/tlrmvm/cpu/TlrmvmCPU.hpp
        )

if(BUILD_DPCPP)
    list(APPEND CPU_SRCS ${PROJECT_SOURCE_DIR}/src/tlrmvm/cpu/TlrmvmDPCPP.cpp)
    list(APPEND CPU_HEADERS ${PROJECT_SOURCE_DIR}/src/tlrmvm/cpu/TlrmvmDPCPP.hpp)
endif()
