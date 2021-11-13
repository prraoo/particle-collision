#include <cstdint>
#include <vector>
#include <math.h>
#include <utils/CUDA/error.h>
#include <iostream>
#include "bloom_kernel.h"
#include <iomanip>
#define SHMEM_SIZE 256

__device__ float average_L;

__constant__ int N;

__device__ float _saturate(float x) {
    return fminf(fmaxf(x, 0), 1);
}

__device__ float brightpass(float pixel, float threshold) {
    return powf(_saturate((pixel - 0.8f * threshold) / 0.2f * threshold), 2);
}

__device__ float tonemap(float pixel) {
    return (pixel * (0.9036f * pixel + 0.018f) / (pixel * (0.8748f * pixel + 0.354f) + 0.14f));
}

__device__ float gamma_correction(float pixel) {
    if (pixel <= 0.0031308f)
        return pixel = 12.92f * pixel;
    else
        return pixel = (1.055f * powf(pixel, 1 / 2.4f)) - 0.055f;
}

__global__ void buffer_kernel(float* Rin_buff, float* Gin_buff, float* Bin_buff, const float* in,
    float mean_luminance, float exposure, float brightpass_threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int outIndex = x;
    int srcIndex = x * 4;

    Rin_buff[outIndex] = in[srcIndex + 0] * exposure * 0.18f / mean_luminance;
    Gin_buff[outIndex] = in[srcIndex + 1] * exposure * 0.18f / mean_luminance;
    Bin_buff[outIndex] = in[srcIndex + 2] * exposure * 0.18f / mean_luminance;

    Rin_buff[outIndex] = brightpass(tonemap(Rin_buff[outIndex]), brightpass_threshold) * in[srcIndex + 0];
    Gin_buff[outIndex] = brightpass(tonemap(Gin_buff[outIndex]), brightpass_threshold) * in[srcIndex + 1];
    Bin_buff[outIndex] = brightpass(tonemap(Bin_buff[outIndex]), brightpass_threshold) * in[srcIndex + 2];
}                        

__global__ void convolution_x_kernel(float* out, float* in, int height, int width, const float* kernel) {

    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    if (Col < width && Row < height) {
        float sum = 0;
        int start = Col - (N / 2);
        for (int j = 0; j < N; j++) {
            int offset = start + j;
            if (offset >= 0 && offset < width) {
                sum += in[Row * width + offset] * kernel[j];
            }
        }
        out[Row * width + Col] = sum;
    }
}

__global__ void convolution_y_kernel(float* out, float* in, int height, int width, const float* kernel) {

    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    if (Col < width && Row < height) {
        float sum = 0;
        int start = Row - (N / 2);
        for (int j = 0; j < N; j++) {
            int offset = start + j;
            if (offset >= 0 && offset < height) {
                sum += in[offset* width + Col] * kernel[j];
            }
        }
        out[Row * width + Col] = sum;
    }
}

__global__ void luminance_kernel(float* out, const float* in) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int outIndex = x;
    int srcIndex = x * 4;
    float R = in[srcIndex + 0];
    float G = in[srcIndex + 1];
    float B = in[srcIndex + 2];

    out[outIndex] = logf(0.2126f * R + 0.7152f * G + 0.0722f * B);
}

__global__ void avg_luminance_kernel(float* avg_luminance, float* luminance) {
    __shared__ float data[SHMEM_SIZE];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    data[threadIdx.x] = luminance[x];
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        if (threadIdx.x % (2 * s) == 0) {
            data[threadIdx.x] += data[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        // avg_luminance[blockIdx.x] = data[0];
        atomicAdd(&average_L, data[0]);
    }
}

__global__ void tonemap_kernel(std::uint32_t* out, const float* in,
                    float* Rout_buff, float* Gout_buff, float* Bout_buff,
                    int width, int height, float mean_luminance, float exposure, float brightpass_threshold){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int outIndex = x;
    int srcIndex = x * 4;
   
    float R = in[srcIndex + 0] * exposure * 0.18f / mean_luminance;
    float G = in[srcIndex + 1] * exposure * 0.18f / mean_luminance;
    float B = in[srcIndex + 2] * exposure * 0.18f / mean_luminance;

    R = (1 - brightpass(tonemap(R), brightpass_threshold)) * in[srcIndex + 0] + Rout_buff[outIndex];
    G = (1 - brightpass(tonemap(G), brightpass_threshold)) * in[srcIndex + 1] + Gout_buff[outIndex];
    B = (1 - brightpass(tonemap(B), brightpass_threshold)) * in[srcIndex + 2] + Bout_buff[outIndex];

    R = gamma_correction(tonemap(R));
    G = gamma_correction(tonemap(G));
    B = gamma_correction(tonemap(B));

    R = (R*255.0f >= 255.0f) ? 255 : lrintf(R*255.0f);
    G = (G*255.0f >= 255.0f) ? 255 : lrintf(G*255.0f);
    B = (B*255.0f >= 255.0f) ? 255 : lrintf(B*255.0f);

    uint32_t R1 = R ;
    uint32_t G1 = G  * 256;
    uint32_t B1 = B  * 256 * 256;
    out[outIndex] = (R1 | G1 | B1);

}

void tonemap(std::uint32_t* out, const float* in, int width, int height, float exposure, float brightpass_threshold)
{
    const int NUM_THREADS = 256;
    const int NUM_BLOCKS = width * height / NUM_THREADS;
    const int SIZE = (height * width) ;
    const int KERNEL_SIZE = 63;

    float avg_luminance = 0;
    float* d_luminance; float* d_block_luminance;
    float* d_bloom_kernel;
    float* d_Rin_buffer; float* d_Gin_buffer; float* d_Bin_buffer;
    float* d_Rout_buffer; float* d_Gout_buffer; float* d_Bout_buffer;
    float* d_Rout1_buffer; float* d_Gout1_buffer; float* d_Bout1_buffer;

    std::vector<float> h_luminance(SIZE);

    float ZERO = 0;
    cudaMemcpyToSymbol(average_L, &ZERO, sizeof(float));
    cudaMemcpyToSymbol(N, &KERNEL_SIZE, sizeof(int));

    // Calculate Luminance
    cudaMalloc(&d_luminance, SIZE * sizeof(float));

    cudaMalloc(&d_Rin_buffer, SIZE * sizeof(float)); cudaMalloc(&d_Rout_buffer, SIZE * sizeof(float)); cudaMalloc(&d_Rout1_buffer, SIZE * sizeof(float));
    cudaMalloc(&d_Gin_buffer, SIZE * sizeof(float)); cudaMalloc(&d_Gout_buffer, SIZE * sizeof(float)); cudaMalloc(&d_Gout1_buffer, SIZE * sizeof(float));
    cudaMalloc(&d_Bin_buffer, SIZE * sizeof(float)); cudaMalloc(&d_Bout_buffer, SIZE * sizeof(float)); cudaMalloc(&d_Bout1_buffer, SIZE * sizeof(float));


    //Calculate Luminance
    luminance_kernel <<< NUM_BLOCKS, NUM_THREADS >>> (d_luminance, in);
    cudaMemcpy(h_luminance.data(), d_luminance, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Calculate Block level average
    cudaMalloc(&d_block_luminance, NUM_BLOCKS * sizeof(float));
    avg_luminance_kernel <<< NUM_BLOCKS, NUM_THREADS >>> (d_block_luminance, d_luminance);
    cudaMemcpyFromSymbol(&avg_luminance, average_L, sizeof(float));
    avg_luminance = exp(avg_luminance / (height * width));

    // Perform Convolution
    cudaMalloc(&d_bloom_kernel, KERNEL_SIZE * sizeof(float));
    cudaMemcpy(d_bloom_kernel, bloom_kernel, KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    buffer_kernel <<< NUM_BLOCKS, NUM_THREADS >>> (d_Rin_buffer, d_Gin_buffer, d_Bin_buffer, in,
                                                   avg_luminance, exposure, brightpass_threshold);

    dim3 dimGrid(ceil(width / 32.0f), ceil(height / 32.0f), 1);
    dim3 dimBlock(32, 32, 1);

    convolution_x_kernel <<< dimGrid, dimBlock >>> (d_Rout_buffer, d_Rin_buffer, height, width, d_bloom_kernel);
    convolution_y_kernel <<< dimGrid, dimBlock >>> (d_Rout1_buffer, d_Rout_buffer, height, width, d_bloom_kernel);
    
    convolution_x_kernel <<< dimGrid, dimBlock >>> (d_Gout_buffer, d_Gin_buffer, height, width, d_bloom_kernel);
    convolution_y_kernel <<< dimGrid, dimBlock >>> (d_Gout1_buffer, d_Gout_buffer, height, width, d_bloom_kernel);
    
    convolution_x_kernel <<< dimGrid, dimBlock >>> (d_Bout_buffer, d_Bin_buffer, height, width, d_bloom_kernel);
    convolution_y_kernel <<< dimGrid, dimBlock >>> (d_Bout1_buffer, d_Bout_buffer, height, width, d_bloom_kernel);
    
    
    // Perform Tone Mapping and Display
    tonemap_kernel<<< NUM_BLOCKS, NUM_THREADS >>>(out, in, 
                                                d_Rout1_buffer, d_Gout1_buffer, d_Bout1_buffer,
                                                width, height, avg_luminance, exposure, brightpass_threshold);
   
    cudaFree(d_luminance);
    cudaFree(d_block_luminance);
    cudaFree(d_Rin_buffer); cudaFree(d_Gin_buffer); cudaFree(d_Bin_buffer);
    cudaFree(d_Rout_buffer);  cudaFree(d_Gout_buffer); cudaFree(d_Bout_buffer);
    cudaFree(d_Rout1_buffer); cudaFree(d_Gout1_buffer); cudaFree(d_Bout1_buffer);
}
