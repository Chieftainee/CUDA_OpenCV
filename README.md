1、opencv用什么build type，就用什么type的库，下载的opencv官方放库就有Debug和Release两种类型

2、默认情况下，VScode下面选择的build type是不会传递给CMAKE_BUILD_TYPE的。
    如果想传递，需要在设置里勾选：在多配置生成器上也设置CMAKE_BUILD_TYPE