#***********************************************************************************#
# CMake是一个构建系统生成器，可以使用单个CMakeLists.txt为不同平台上的不同工具集配置项目。
# 您可以在CMakeLists.txt中描述构建系统必须运行的操作，以配置并编译代码。基于这些指令，
# CMake将为所选的构建系统(Unix Makefile、Ninja、Visual Studio等等)生成相应的指令。
# 命令行方式: cmake -G Ninja ..，用`-G`切换生成器。
#***********************************************************************************#

#-----------------------------------------------------------------------------------#
# 外部构建
# 
# Disable in-source builds to prevent source tree corruption.
#-----------------------------------------------------------------------------------#
if(" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "FATAL: In-source builds are not allowed.
        You should create a separate directory for build files.")
endif()

#-----------------------------------------------------------------------------------#
# 声明项目的名称和支持的编程语言(CXX代表C++)
# 
# 使得CMake可以识别和验证它需要的编译器，并缓存结果。
# CMAKE变量 CMAKE_CUDA_STANDARD 指定CUDA C++标准
#-----------------------------------------------------------------------------------#
cmake_minimum_required (VERSION 3.18 FATAL_ERROR)
project (CUDA_CS344 VERSION 1.1 LANGUAGES CXX CUDA)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "Configuring on/for Linux")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message(STATUS "Configuring on/for Windows")
else()
    message(STATUS "Configuring on/for ${CMAKE_SYSTEM_NAME}")
endif()


#-----------------------------------------------------------------------------------#
# option定义一个USE_*变量当做宏开关
# 
# configure_file 读取输入文件中的内容，将 CMakeLists.txt 文件中的变量转变为 C/C++ 中可识别的宏定义，存入另一个文件中
#-----------------------------------------------------------------------------------#
option(USE_01_Lession "Use 01_Lession convert image rgba to grey implementation" ON)
configure_file(Config.h.in Config.h)


#-----------------------------------------------------------------------------------#
# 如果CMAKE_CXX_STANDARD_REQUIRED设置为True，则必须使用CMAKE_CXX_STANDARD指定的版本，
# 如果CMAKE_CXX_STANDARD_REQUIRED设置为OFF,  则CMAKE_CXX_STANDARD指定版本的为首选版本，如果没有会使用上一版本。
#-----------------------------------------------------------------------------------#
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#-----------------------------------------------------------------------------------#
# 查找CUDAToolkit包 和 OpenCV包
# 
# 使得CMake可以识别和验证它需要的编译器，并缓存结果。
# 注意OpenCV编译的库需要和生成器匹配：这里使用的kit是Visual Studio Community 2017 Release - x86_amd64
#-----------------------------------------------------------------------------------#
find_package(CUDAToolkit REQUIRED)
message(STATUS "Found CUDA-${CUDAToolkit_VERSION} at ${CUDAToolkit_LIBRARY_ROOT}")


# 设置路径寻找OpenCVConfig.cmake文件
set(OpenCV_DIR "C:/E_File_Disk/OpenCV/OpenCV/OpenCV-4.5.4-CUDA/opencv/build")
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV_DIR          = ${OpenCV_DIR}")
message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS         = ${OpenCV_LIBS}")

include_directories(${OpenCV_INCLUDE_DIRS})

#-----------------------------------------------------------------------------------#
# 'CMAKE_BUILD_TYPE' 变量展示了如何切换生成构建系统的配置, 设置默认的构建类型(本例中是Debug)
#-----------------------------------------------------------------------------------#
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
    message(STATUS "Debug build ... ")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    message(STATUS "Debug build ... ")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RELEASE")
    message(STATUS "Release build ... ")
else()
    message(FATAL_ERROR "FATAL: invalid CMAKE_BUILD_TYPE.")
endif()

#-----------------------------------------------------------------------------------#
# 创建目标——静态库
#
# 'add_subdirectory(01_Lession)' 将子目录 01_Lession（含CMakeList.txt） 里的文件编译成静态库
#-----------------------------------------------------------------------------------#
if(USE_Lession-01)
    add_subdirectory(src/01_Lession)
    list(APPEND EXTRA_LIBS 01_Lession)
    list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/include/01_Lession")
endif()


#-----------------------------------------------------------------------------------#
# 创建可执行目标文件
#-----------------------------------------------------------------------------------#
add_executable(${CMAKE_PROJECT_NAME} test/main.cpp)

#-----------------------------------------------------------------------------------#
# 最后，将目标库链接到可执行目标
#
# target_include_directories()：指定目标包含的头文件路径
# target_link_libraries()：     指定目标链接的库
# target_compile_options()：    指定目标的编译选项
#-----------------------------------------------------------------------------------#
# configure_file生成的Config.h文件就在build目录下, 所以include加了${PROJECT_BINARY_DIR}路径
target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${EXTRA_INCLUDES} ${PROJECT_BINARY_DIR})
target_link_libraries(${CMAKE_PROJECT_NAME} ${EXTRA_LIBS} ${OpenCV_LIBS})

# GLOB_RECURSE 是递归查找
# file(GLOB INCLUDE_FILES *.h *.hpp *.cuh)
# file(GLOB CPP_FILES *.c *.cpp *.cu)

#add_library(rgba_to_grey STATIC rgba_to_grey.cu)
# add_library(testlib STATIC rgba_to_grey.cu)

# nvcc使用CUDA C++11 build target
# As this is a public compile feature anything that
# links to rgba_to_grey will also build with -std=c++11
# 也可以用  CMAKE_CUDA_STANDARD 
# target_compile_features(testlib PUBLIC cxx_std_11)

# 设置compiler flag fPIC，不幸的是，并不是所有编译器都支持fPIC命令，
# 所以CMake提供了构建shared library设置target与位置无关的属性
# set_target_properties(rgba_to_grey PROPERTIES POSITION_INDEPENDENT_CODE ON)

# We need to explicitly state that we need all CUDA files in the 
# rgba_to_grey library to be built with -dc as the member functions 
# could be called by other libraries and executables
# 分离编译，一种增量构建的方式只允许重新编译和链接已修改的单元，这减少了构建时间
# set_target_properties( testlib
#                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON
#                       )

# set(CUDA_NVCC_FLAGS -arch=compute_50;-code=sm_50)

# CMAKE 3.12
# target_compile_options(target PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#                      --generate-line-info
#                       --use_fast_math
#                      --relocatable-device-code=true
#                       >)

# 文件归类到include 和 src 文件夹中
# source_group("include" FILES ${INCLUDE_FILES})
# source_group("src" FILES ${CPP_FILES})


# set(extra_libs "")
# 
# if(HAVE_CUBLAS)
#   list(APPEND extra_libs ${CUDA_cublas_LIBRARY})
# endif()
# 
# if(HAVE_CUFFT)
#   list(APPEND extra_libs ${CUDA_cufft_LIBRARY})
# endif()
# 
# 
# target_link_libraries(test cuda.lib cudart_static.lib)
# 
# if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#     set(CMAKE_CUDA_ARCHITECTURES 75)
#   endif()
# 
# 
# set(CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50;-G;-g)
# 
# #set(CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20;-G;-g)
# #set(CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52;-G;-g)
# 
# cuda_add_library(gpu rgba_to_grey.cu)
# 
# CMAKE_CUDA_FLAGS=”-arch=sm_30” .passes -arch=sm_30 to nvcc, 
# telling it to target the Kepler architecture (SM_30 or Compute Capability 3.0) GPU in my computer.
# 
# Next, Figure 1 shows how I invoked the build with the command make -j4. 
# This runs make with multiple threads so it compiles the C++ and CUDA source files in parallel.
# 
# You can specify which runtime all CUDA targets will use inside your project by setting CMAKE_CUDA_FLAGS
#  to –cudart { none | shared | static }. For example, to use the static CUDA runtime library, 
#  set it to –cudart static.
