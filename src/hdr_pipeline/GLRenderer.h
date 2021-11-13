#ifndef INCLUDED_GLRENDERER
#define INCLUDED_GLRENDERER

#pragma once

#include <cstdint>
#include <string>
#include <optional>
#include <chrono>

#include <GL/gl.h>

#include <GL/platform/Renderer.h>
#include <GL/platform/Context.h>
#include <GL/platform/Window.h>
#include <GL/platform/DefaultDisplayHandler.h>

#include <GL/shader.h>
#include <GL/texture.h>
#include <GL/framebuffer.h>
#include <GL/vertex_array.h>

#include <cuda_runtime_api.h>

#include <utils/CUDA/event.h>
#include <utils/CUDA/graphics_interop.h>

#include <utils/image.h>

#include "HDRPipeline.h"


class GLScene;

class GLRenderer : public virtual GL::platform::Renderer, private GL::platform::DefaultDisplayHandler
{
	GL::platform::Window window;
	GL::platform::Context context;
	GL::platform::context_scope<GL::platform::Window> ctx;

	const GLScene* scene = nullptr;
	std::optional<HDRPipeline> pipeline;

	int framebuffer_width;
	int framebuffer_height;

	GL::Texture hdr_buffer;
	GL::Texture ldr_buffer;
	GL::Renderbuffer depth_buffer;

	GL::Framebuffer fbo;

	GL::VertexArray vao;

	GL::Program fullscreen_prog;

	GL::Sampler sampler;

	CUDA::graphics::unique_resource hdr_buffer_resource;
	CUDA::graphics::unique_resource ldr_buffer_resource;

	float exposure;
	float brightpass_threshold;

	long frame_count = 0;

	std::string title;

	std::chrono::steady_clock::time_point next_fps_tick = std::chrono::steady_clock::now();

	float pipeline_time = 0.0f;
	CUDA::unique_event pipeline_begin = CUDA::create_event();
	CUDA::unique_event pipeline_end = CUDA::create_event();

	void resize(int width, int height, GL::platform::Window*) override;

public:
	GLRenderer(std::string title, int width, int height, float exposure, float brightpass_threshold);

	image2D<std::uint32_t> screenshot() const;

	void attach(GL::platform::MouseInputHandler* mouse_input);
	void attach(GL::platform::KeyboardInputHandler* keyboard_input);
	void attach(const GLScene* scene);

	void render() override;
};

#endif  // INCLUDED_GLRENDERER
