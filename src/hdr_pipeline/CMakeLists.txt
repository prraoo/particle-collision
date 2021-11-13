cmake_minimum_required(VERSION 3.18)

if (CUDAToolkit_FOUND)

project(hdr_pipeline CXX CUDA)

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(hdr_pipeline
	"${SOURCE_DIR}/envmap.cpp"
	"${SOURCE_DIR}/envmap.h"
	"${SOURCE_DIR}/HDRPipeline.h"
	"${SOURCE_DIR}/HDRPipeline.cpp"
	"${SOURCE_DIR}/hdr_pipeline.cu"
	"${SOURCE_DIR}/HDRDemo.h"
	"${SOURCE_DIR}/HDRDemo.cpp"
	"${SOURCE_DIR}/main.cpp"
)

if (INTERACTIVE)
	target_sources(hdr_pipeline PRIVATE
		"${SOURCE_DIR}/GLSL/envmap"
		"${SOURCE_DIR}/GLSL/fullscreen_triangle.vs.glsl"
		"${SOURCE_DIR}/GLSL/fullscreen_triangle.fs.glsl"
		"${SOURCE_DIR}/GLSL/env.vs.glsl"
		"${SOURCE_DIR}/GLSL/env.fs.glsl"
		"${SOURCE_DIR}/GLSL/model.vs.glsl"
		"${SOURCE_DIR}/GLSL/model.fs.glsl"
		"${SOURCE_DIR}/InputHandler.h"
		"${SOURCE_DIR}/InputHandler.cpp"
		"${SOURCE_DIR}/GLRenderer.h"
		"${SOURCE_DIR}/GLRenderer.cpp"
		"${SOURCE_DIR}/GLScene.h"
		"${SOURCE_DIR}/GLScene.cpp"
		"${SOURCE_DIR}/run_interactive.cpp"
	)
else ()
	target_sources(hdr_pipeline PRIVATE
		"${SOURCE_DIR}/run_noninteractive.cpp"
	)
endif ()

configure_project(hdr_pipeline)

if (INTERACTIVE)
	target_link_libraries(hdr_pipeline_shaders GLSL_utils)
endif ()

target_link_libraries(hdr_pipeline utils CUDA_utils)

endif ()
