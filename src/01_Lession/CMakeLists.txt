# if(IOS OR WINRT OR (NOT HAVE_CUDA))
#     message(FATAL_ERROR "FATAL: No Support CUDA")
# endif()

set(the_description "CUDA-accelerated Image Processing")


# # 'CUDA_ARCHITECTURES' is initialized by the value of the CMAKE_CUDA_ARCHITECTURES variable
# if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#   set(CMAKE_CUDA_ARCHITECTURES 75)
# endif()

# GLOB_RECURSE 是递归查找
# file(GLOB INCLUDE_FILES *.hpp)
file(GLOB CPP_FILES *.cu)

add_library(01_Lession STATIC ${CPP_FILES})

#-----------------------------------------------------------------------------------#
# set_target_properties(tgt PROPERTIES CUDA_ARCHITECTURES "35;50;72")
# Generates code for real and virtual architectures 30, 50 and 72.
# 
# set_property(TARGET tgt PROPERTY CUDA_ARCHITECTURES 70-real 72-virtual)
# Generates code for real architecture 70 and virtual architecture 72.
# 
# set_property(TARGET tgt PROPERTY CUDA_ARCHITECTURES OFF)
# CMake will not pass any architecture flags to the compiler.
#-----------------------------------------------------------------------------------#
set_target_properties(01_Lession PROPERTIES CUDA_ARCHITECTURES "50")