cmake_minimum_required(VERSION 3.18)
project(TLRMVMpy LANGUAGES CXX)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
pybind11_add_module(${PROJECT_NAME} src/wrapper.cpp)
if(USE_MKL)
target_link_libraries(${PROJECT_NAME} PRIVATE tlrmvmcpulib ${TLRMVM_LIBS} -lmkl_rt -lmkl_def)
else()
target_link_libraries(${PROJECT_NAME} PRIVATE tlrmvmcpulib ${TLRMVM_LIBS})
endif()
target_include_directories(${PROJECT_NAME} PRIVATE ${TLRMVM_INCS})
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION python/tlrmvmpy)

install(FILES 
${CMAKE_CURRENT_LIST_DIR}/__init__.py 
${CMAKE_CURRENT_LIST_DIR}/tlrmvmtools.py 
${CMAKE_CURRENT_LIST_DIR}/tlrmat.py 
${CMAKE_CURRENT_LIST_DIR}/wrapper.py
DESTINATION python/tlrmvmpy)
