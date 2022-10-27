#include <iostream>
#include <stdio.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaarithm.hpp"

#include "Config.h"


// 头文件感应是VScode提供的，但是编译路径在cmake里给的
#ifdef USE_01_Lession
    #include "rgba_to_grey.h"
#endif

using namespace std;
using namespace cv;

// check device
bool deviceInitialize();

int main(int argc, char **argv)
{
    
    // report version
    std::cout << endl;
    std::cout << argv[0] << " Version " << CUDA_CS344_VERSION_MAJOR << "." << CUDA_CS344_VERSION_MINOR << std::endl;

    // device 初始化
    deviceInitialize();

    // 读取源图像
    Mat image;
    image = imread("C:/Users/45487/Desktop/CUDA_OpenCV/resources/cinque_terre_small.jpg", IMREAD_COLOR);
    if (image.empty())
    {
        printf("load failure");
        exit(1);
    }
    imshow("image", image);

    
    // create 
    Mat imageRGBA, imageGrey;
    imageGrey.create(image.rows, image.cols, CV_8UC1);
    cvtColor(image, imageRGBA, COLOR_BGR2BGRA);

    if (!imageRGBA.isContinuous() || !imageGrey.isContinuous())
    {
        std::cerr << "Images aren't continuous!! Exiting." << std::endl;
        exit(1);
    }
    imshow("imageRGBA", imageRGBA);

    // convert to grey
    cuda::GpuMat src, dst;
    src.upload(imageRGBA);
    gpu_rgba_to_greyscale(src, dst);
    dst.download(imageGrey);
    imshow("imageGrey", imageGrey);

    waitKey(0);
    return 0;
}


// check device
bool deviceInitialize()
{
      // 获取设备的数量
    int num_devices = cuda::getCudaEnabledDeviceCount();
    if (num_devices <= 0)
    {
        std::cerr << "There is no device." << std::endl;
        exit(1);
    }
    // 计算可用设备的数量
    int enable_device_id = -1;
    for (int i = 0; i < num_devices; i++)
    {
        cuda::DeviceInfo dev_info(i);
        if (dev_info.isCompatible())
        {
            enable_device_id = i;
        }
    }
    // 检测可用设备的数量，保证至少有一个可用设备
    if (enable_device_id < 0)
    {
        std::cerr << "GPU module isn't built for GPU" << std::endl;
        exit(1);
    }

    // 打印设备信息
    cuda::printCudaDeviceInfo(enable_device_id);
    cuda::setDevice(enable_device_id);
    std::cout << "GPU is ready, device ID is " << num_devices << "\n";

    // There is a set of methods to check whether the module contains intermediate (PTX) or 
    // binary CUDA code for the given architecture(s)
    printf("Whether the module contains intermediate(PTX) or binary CUDA code for the given architecture(s) : %s\n",
        cuda::TargetArchs::has(5, 0) ? "Yes" : "No");
    printf("Whether the module contains intermediate(PTX) CUDA code for the given architecture(s) :           %s\n",
        cuda::TargetArchs::hasPtx(5, 0) ? "Yes" : "No");
    printf("Whether the module contains binary CUDA code for the given architecture(s) :                      %s\n",
        cuda::TargetArchs::hasBin(5, 0) ? "Yes" : "No");
    printf("Whether the module contains intermediate(PTX) CUDA code for the given architecture(s) :           %s\n",
        cuda::TargetArchs::hasPtx(6, 0) ? "Yes" : "No");

    return EXIT_SUCCESS;
}
