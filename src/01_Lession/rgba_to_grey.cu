#include <cuda_runtime.h>
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"

#define CHECK(call)											   				\
{															   				\
	const cudaError_t error = call;							   				\
	if (error != cudaSuccess)								   				\
	{														   				\
		fprintf(stderr, "Error: ##########################\n");				\
		fprintf(stderr, "FILE: %s\n", __FILE__);               				\
		fprintf(stderr, "LINE: %d\n", __LINE__);               				\
		fprintf(stderr, "CUDA Code: %d, reason: %s\n",		   				\
			error, cudaGetErrorString(error));				   				\
		fprintf(stderr, "##########################: Error\n");				\
		exit(1);											   				\
	}														   				\
}

__global__ 
void kernel(const cv::cuda::PtrStepSz<uchar4> src, cv::cuda::PtrStep<uchar1> dst)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if(ix < src.cols && iy < src.rows)
    {
        uchar4 color = src(iy, ix);
        dst(iy, ix) = make_uchar1(0.299f * color.x + 0.587f * color.y + 0.114f * color.z);
    }
}

// void gpu_rgba_to_greyscale(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::Stream& stream)
void gpu_rgba_to_greyscale(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst)
{
    // kernel函数配置
    dim3 block(16, 16);
    dim3 grid((src.cols + block.x - 1) / block.x, (src.rows + block.y - 1) / block.y);
    printf("cuda kernel grid  = (%d, %d, %d)\n", grid.x, grid.y, grid.z);
    printf("cuda kernel block = (%d, %d, %d)\n", block.x, block.y, block.z);

    dst.create(src.size(), CV_8UC1);
    kernel<<<grid, block>>>(src, dst);
    CHECK(cudaGetLastError());

    // Class StreamAccessor that enables getting cudaStream_t from cuda::Stream
    // cudaStream_t s = cv::cuda::StreamAccessor::getStream(stream);
    cudaDeviceSynchronize();
}