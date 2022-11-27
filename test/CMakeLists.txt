###################
# CPU Test build
###################
if (BUILD_CPU)
    set(BINS
            Test_cpu_tlrmvm
            )
    foreach(cbins ${BINS})
        WrapBinary(${cbins} cpp)
        install(TARGETS ${cbins} DESTINATION test)
    endforeach()
    if(USE_MPI)
        set(BINS
                ex2mpitlrmvm_float
                ex2mpitlrmvm_complexfloat
                ex3_gendata
                )
        foreach(cbins ${BINS})
            WrapBinary(${cbins} cpp)
            install(TARGETS ${cbins} DESTINATION test)
        endforeach()
    endif() # USE MPI
endif()

###################
# DPCPP Test build
###################
if (BUILD_DPCPP)
    WrapBinary(ex4oneapi oneapi)
endif()

###################
# GPU Test build
###################
if(BUILD_CUDA)
    set(BINS
            Test_BatchCall_Tlrmvm
            )
    if(USE_MPI)
    LIST(APPEND BINS Test_MultiGPUBatchCall_Tlrmvm)
    endif()
    foreach(cbins ${BINS})
        WrapCUDABinary(${cbins} cuda)
        install(TARGETS ${cbins} DESTINATION test)
    endforeach()
endif()

if(BUILD_HIP)

    set(BINS
            Test_hip_hipblas
            Test_hip_tlrmvm
            Test_hip_constrank
#            Test_hip_tlrmvmgraph
            )
    foreach(cbins ${BINS})
        WrapHIPBinary(${cbins} hip)
        install(TARGETS ${cbins} DESTINATION test)
    endforeach()
#    install(FILES
#            ${CMAKE_CURRENT_LIST_DIR}/hip/Test_hip_tlrmvm_correctness.sh
#            DESTINATION test PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE)
endif()