#ifndef INCLUDED_RENDERER
#define INCLUDED_RENDERER

#pragma once

#include <cstdint>
#include <chrono>

#include <GL/gl.h>

#include <GL/platform/Renderer.h>
#include <GL/platform/Context.h>
#include <GL/platform/Window.h>
#include <GL/platform/DefaultDisplayHandler.h>

#include <utils/image.h>

#include "GLScene.h"


class GLRenderer : public virtual GL::platform::Renderer, private GL::platform::DefaultDisplayHandler
{
	GL::platform::Window window;
	GL::platform::Context context;
	GL::platform::context_scope<GL::platform::Window> ctx;

	int framebuffer_width;
	int framebuffer_height;

	GLScene* scene = nullptr;

	bool frozen = true;

	std::chrono::nanoseconds timestep;
	int max_frames;

	int step_count = 0;
	long long frame_count = 0;

	std::chrono::steady_clock::time_point last_update;
	std::chrono::steady_clock::time_point next_fps_tick = std::chrono::steady_clock::now();

	std::chrono::nanoseconds simulation_time;

	float update_time = 0.0f;

	bool show_fps = false;

	void resize(int width, int height, GL::platform::Window*) override;
	void update_window_title(float dt);

public:
	GLRenderer(int max_frames, float dt, bool frozen);

	void reset();

	bool toggle_freeze();
	bool toggle_fps();

	void step(int steps = 1);

	void render() override;

	image2D<std::uint32_t> screenshot() const;

	void attach(GL::platform::MouseInputHandler* mouse_input);
	void attach(GL::platform::KeyboardInputHandler* keyboard_input);
	void attach(GLScene* scene);
};

#endif  // INCLUDED_RENDERER
