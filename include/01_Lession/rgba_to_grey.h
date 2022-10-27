#ifndef __RGBA_TO_GREY_CUH__
#define __RGBA_TO_GREY_CUH__

#include "opencv2/core/cuda.hpp"

// void gpu_rgba_to_greyscale(const cv::cuda::GpuMat &src, cv::cuda::GpuMat& dst, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
void gpu_rgba_to_greyscale(const cv::cuda::GpuMat &src, cv::cuda::GpuMat& dst);

#endif