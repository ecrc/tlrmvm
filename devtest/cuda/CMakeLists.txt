cmake_minimum_required(VERSION 3.18)
set(CUDA_TESTS SeismicCUDAGraph)

foreach(exec ${CUDA_TESTS})
    add_executable(${exec} ${CMAKE_CURRENT_LIST_DIR}/${exec}.cpp)
    target_link_libraries(${exec} PUBLIC tlrmvmcudalib benchmark::benchmark gtest_main gmock_main)

    if(USE_NVTX)
    add_executable(${exec}_nvtx ${CMAKE_CURRENT_LIST_DIR}/${exec}.cpp)
    target_link_libraries(${exec}_nvtx PUBLIC tlrmvmcudalib benchmark::benchmark gtest_main gmock_main)
    target_compile_definitions(${exec}_nvtx PUBLIC USE_NVTX)
    target_link_libraries(${exec}_nvtx PUBLIC nvToolsExt)
    endif()
endforeach()


foreach(exec ${CUDA_TESTS})
install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${exec}
    DESTINATION test/cuda
    PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE
)
endforeach()