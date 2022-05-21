cmake_minimum_required(VERSION 3.19)
project(TLRMVMpy LANGUAGES CXX)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
pybind11_add_module(${PROJECT_NAME}
        src/Wrapper.cpp
        src/Tlrmvm.cpp
        src/BatchTlrmvm.cpp
        src/CommonWrapper.cpp
        )

set(INCLIBS)
if(BUILD_CPU)
    set(INCLIBS tlrmvmcpulib)
endif()
if(BUILD_CUDA)
    set(INCLIBS tlrmvmcudalib)
endif()
if(BUILD_HIP)
    set(INCLIBS tlrmvmhiplib)
    message(${INCLIBS})
endif()

target_include_directories(${PROJECT_NAME} PRIVATE ../src)
target_link_libraries(${PROJECT_NAME} PRIVATE ${INCLIBS} ${TLRMVM_LIBS})
target_include_directories(${PROJECT_NAME} PRIVATE ${TLRMVM_INCS})

install(TARGETS ${PROJECT_NAME} LIBRARY
        DESTINATION python/tlrmvmpy
        PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE)

install(FILES
        ${CMAKE_CURRENT_LIST_DIR}/__init__.py
        ${CMAKE_CURRENT_LIST_DIR}/tlrmvmtools.py
        ${CMAKE_CURRENT_LIST_DIR}/tlrmat.py
        ${CMAKE_CURRENT_LIST_DIR}/_wrapper.py
        DESTINATION python/tlrmvmpy
        PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE)
