set(DIR_SRCS)

file(GLOB TMP *.cpp)
list(APPEND DIR_SRCS ${TMP})
file(GLOB TMP vendorblas/*.cpp)
list(APPEND DIR_SRCS ${TMP})
set(TLRMVM_SRCS ${TLRMVM_SRCS} ${DIR_SRCS} PARENT_SCOPE)
