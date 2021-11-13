cmake_minimum_required(VERSION 3.18)

if (CUDAToolkit_FOUND) 
	project(particles CXX CUDA)
else ()
	project(particles CXX)
endif ()

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")


add_library(particle_system_state
	"${SOURCE_DIR}/particle_system_state.h"
	"${SOURCE_DIR}/particle_system_state.cpp"
)

configure_project(particle_system_state)

target_include_directories(particle_system_state PUBLIC "${SOURCE_DIR}")
target_link_libraries(particle_system_state PUBLIC utils)


if (CUDAToolkit_FOUND)
	add_library(particle_system SHARED
		"${SOURCE_DIR}/particles.cu"
		"${SOURCE_DIR}/ParticleSystem.h"
		"${SOURCE_DIR}/ParticleSystem.cpp"
		"${SOURCE_DIR}/particle_system_module.h"
		"${SOURCE_DIR}/particle_system_module.cpp"
	)
	configure_project(particle_system)
	target_link_libraries(particle_system PRIVATE CUDA_utils)
	set_target_properties(particle_system PROPERTIES PREFIX "")
endif ()


add_executable(particles
	"${SOURCE_DIR}/ParticleSystemLoader.h"
	"${SOURCE_DIR}/ParticleDemo.h"
	"${SOURCE_DIR}/main.cpp"
)

if (INTERACTIVE)
	target_sources(particles PRIVATE
		"${SOURCE_DIR}/GLSL/particle.vs.glsl"
		"${SOURCE_DIR}/GLSL/particle.gs.glsl"
		"${SOURCE_DIR}/GLSL/particle.fs.glsl"
		"${SOURCE_DIR}/GLSL/bounding_box.vs.glsl"
		"${SOURCE_DIR}/GLSL/bounding_box.fs.glsl"
		"${SOURCE_DIR}/InputHandler.h"
		"${SOURCE_DIR}/InputHandler.cpp"
		"${SOURCE_DIR}/GLParticlePipeline.h"
		"${SOURCE_DIR}/GLParticlePipeline.cpp"
		"${SOURCE_DIR}/GLBoundingBox.h"
		"${SOURCE_DIR}/GLBoundingBox.cpp"
		"${SOURCE_DIR}/GLScene.h"
		"${SOURCE_DIR}/GLScene.cpp"
		"${SOURCE_DIR}/GLParticleReplay.h"
		"${SOURCE_DIR}/GLParticleReplay.cpp"
		"${SOURCE_DIR}/GLRenderer.h"
		"${SOURCE_DIR}/GLRenderer.cpp"
		"${SOURCE_DIR}/GLParticleDemo.h"
		"${SOURCE_DIR}/GLParticleDemo.cpp"
	)
endif ()

if (CUDAToolkit_FOUND)
	if (INTERACTIVE)
		target_sources(particles PRIVATE
			"${SOURCE_DIR}/GLCUDAParticles.h"
			"${SOURCE_DIR}/GLCUDAParticles.cpp"
			"${SOURCE_DIR}/GLCUDAParticleDemo.cpp"
		)
	else ()
		target_sources(particles PRIVATE
			"${SOURCE_DIR}/CUDAParticles.h"
			"${SOURCE_DIR}/CUDAParticles.cpp"
			"${SOURCE_DIR}/NoGLCUDAParticleDemo.cpp"
		)
	endif ()
	target_link_libraries(particles CUDA_utils)
	add_dependencies(particles particle_system)
else ()
	if (INTERACTIVE)
		target_sources(particles PRIVATE
			"${SOURCE_DIR}/GLNoCUDAParticleDemo.cpp"
		)
	else ()
		message(FATAL_ERROR "cannot build particles in neither interactive mode nor with CUDA")
	endif ()
endif ()

configure_project(particles)
target_link_libraries(particles particle_system_state utils)

if (INTERACTIVE)
	target_link_libraries(particles_shaders GLSL_utils)
endif ()
