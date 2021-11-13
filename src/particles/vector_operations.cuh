#include "particle_system_module.h"
#include <utils/CUDA/error.h>
#pragma once

inline __device__ float3 operator * (float3 a, float b) {
	return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __device__ float3 operator * (float a, float3 b) {
	return make_float3(a * b.x, a * b.y, a * b.z);
}

inline __device__ float3 operator - (float3 a, float3 b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ float3 operator + (float3 a, float3 b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ float3 operator / (float3 a, float b) {
	return make_float3(a.x / b, a.y / b, a.z / b);
}

inline __device__ void operator += (float3& a, float3 b) {
	a.x += b.x; a.y += b.y; a.z += b.z;
}

inline __device__ float3 make_float3(float4 a) {
	return make_float3(a.x, a.y, a.z);
}
