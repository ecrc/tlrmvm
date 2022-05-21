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
