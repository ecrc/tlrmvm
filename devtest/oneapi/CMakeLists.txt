cmake_minimum_required(VERSION 3.21)
set(ONEAPI_TESTS Astronomy SeismicRedatuming ixpug)

foreach(exec ${ONEAPI_TESTS})

add_executable(${exec} ${CMAKE_CURRENT_LIST_DIR}/${exec}.cpp)
target_include_directories(${exec} PRIVATE ${TLRMVM_INCS})
target_link_libraries(${exec} PUBLIC tlrmvmcpulib benchmark::benchmark gtest_main gmock_main)


endforeach(exec)

foreach(test ${ONEAPI_TESTS})
install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${test}
    DESTINATION bin/testing/oneapi
    PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE
)
endforeach()

# add_executable(ixpug ${CMAKE_CURRENT_LIST_DIR}/ixpug.cpp)
# target_include_directories(ixpug PUBLIC ${TLRMVM_INCS})
# target_link_libraries(ixpug PUBLIC ${TLRMVM_LIBS})