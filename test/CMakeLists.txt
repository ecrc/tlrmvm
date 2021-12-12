
#dpcpp build
if(BUILD_DPCPP)
WrapBinary(ex4oneapi oneapi)
else()
WrapBinary(ex1basic_float cpp)
WrapBinary(ex1basic_complexfloat cpp)
install(FILES ${CMAKE_CURRENT_LIST_DIR}/cpp/runexp1.sh DESTINATION test
PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE)
install(TARGETS ex1basic_float DESTINATION test)
install(TARGETS ex1basic_complexfloat DESTINATION test)
if(USE_MPI)
WrapBinary(ex2mpitlrmvm_float cpp)
WrapBinary(ex2mpitlrmvm_complexfloat cpp)
WrapBinary(ex3_gendata cpp)
install(TARGETS ex3_gendata DESTINATION test)
install(TARGETS ex2mpitlrmvm_float DESTINATION test)
install(TARGETS ex2mpitlrmvm_complexfloat DESTINATION test)
install(FILES 
${CMAKE_CURRENT_LIST_DIR}/cpp/runexp2AMD.sh 
${CMAKE_CURRENT_LIST_DIR}/cpp/runexp2AMD_synthetic.sh 
${CMAKE_CURRENT_LIST_DIR}/cpp/runexp2ICX.sh 
${CMAKE_CURRENT_LIST_DIR}/cpp/runexp2NEC.sh 
${CMAKE_CURRENT_LIST_DIR}/cpp/runexp2A64FX.sh 
${CMAKE_CURRENT_LIST_DIR}/cpp/runexp3_gendata.sh
DESTINATION test
PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE)
endif() # USE MPI

endif() # DPCPP
if(BUILD_CUDA)
WrapCUDABinary(ex3basic cuda)
WrapCUDABinary(ex4cudagraph_float cuda)
WrapCUDABinary(ex4cudagraph_csingle cuda)
WrapCUDABinary(ex5dense cuda)
install(TARGETS 
ex3basic ex4cudagraph_csingle ex4cudagraph_float ex5dense
DESTINATION test)
install(FILES ${CMAKE_CURRENT_LIST_DIR}/cuda/cudatest.sh DESTINATION test
PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE)
endif()

install(FILES 
${CMAKE_CURRENT_LIST_DIR}/python/tlrmvmexample.py 
${CMAKE_CURRENT_LIST_DIR}/python/generateinput.py 
${CMAKE_SOURCE_DIR}/python/tlrmvmtools.py 
${CMAKE_CURRENT_LIST_DIR}/python/pyexec.sh
DESTINATION test/python
PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE)