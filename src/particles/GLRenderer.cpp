#include <cstdint>
#include <memory>
#include <string_view>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <iomanip>

#include <GL/error.h>

#include "GLRenderer.h"

using namespace std::literals;


#ifdef WIN32
extern "C" __declspec(dllexport) DWORD NvOptimusEnablement = 1U;
#endif

namespace
{
#ifdef WIN32
	void APIENTRY debug_output_callback(GLenum /*source*/, GLenum /*type*/, GLuint /*id*/, GLenum /*severity*/, GLsizei /*length*/, const GLchar* message, const void* /*userParam*/)
	{
		OutputDebugStringA(message);
		OutputDebugStringA("\n");
	}
#else
	void debug_output_callback(GLenum /*source*/, GLenum /*type*/, GLuint /*id*/, GLenum /*severity*/, GLsizei /*length*/, const GLchar* message, const void* /*userParam*/)
	{
		std::cerr << message << '\n';
	}
#endif
}

GLRenderer::GLRenderer(int max_frames, float dt, bool frozen)
	: window("particles", 1024, 768, 24),
	  context(window.createContext(4, 3, true)),
	  ctx(context, window),
	  max_frames(max_frames),
	  frozen(frozen),
	  timestep(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<float>(dt)))
{
	std::cout << "\nOpenGL on " << glGetString(GL_RENDERER) << '\n' << std::flush;

	glDebugMessageCallback(debug_output_callback, nullptr);
	glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);


	glEnable(GL_FRAMEBUFFER_SRGB);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	glClearColor(0.6f, 0.7f, 1.0f, 1.0f);
	glClearDepth(1.0f);

	ctx.setSwapInterval(0);

	window.attach(this);
	GL::throw_error();
}

void GLRenderer::resize(int width, int height, GL::platform::Window*)
{
	framebuffer_width = width;
	framebuffer_height = height;
}

void GLRenderer::render()
{
	if (framebuffer_width <= 0 || framebuffer_height <= 0)
		return;

	glViewport(0, 0, framebuffer_width, framebuffer_height);

	auto now = std::chrono::steady_clock::now();

	if (scene && !frozen)
	{
		std::chrono::nanoseconds dt = now - last_update;

		if (dt > timestep)
		{
			step(dt / timestep);
			last_update = now;
		}
	}

	glDepthMask(GL_TRUE);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (scene)
	{
		scene->draw(framebuffer_width, framebuffer_height);
	}


	++frame_count;

	if (next_fps_tick <= now)
	{
		constexpr auto fps_update_timestep = 100ms;

		auto dt = std::chrono::duration<float>(now - next_fps_tick + fps_update_timestep).count();

		update_window_title(dt);

		next_fps_tick = now + fps_update_timestep;
		frame_count = 0;
	}

	GL::throw_error();

	ctx.swapBuffers();
}

void GLRenderer::update_window_title(float dt)
{
	std::ostringstream str;
	str << "particles @ t_avg = "sv
	    << std::setprecision(3) << std::fixed << update_time / step_count << " ms    "sv
	    << std::setw(static_cast<int>(std::log10(max_frames)) + 1) << step_count << " simulation steps * "sv
	    << std::defaultfloat << std::chrono::duration<double>(timestep).count() << " s = "sv
	    << std::fixed << std::setw(static_cast<int>(std::log10(std::chrono::duration<double>(max_frames * timestep).count())) + 4) << std::chrono::duration<double>(simulation_time).count() << " s";

	if (show_fps)
		str << "    "sv << std::setprecision(1) << std::fixed << dt * 1000.0f / frame_count << " mspf = "sv << frame_count / dt << " fps"sv;

	window.title(str.str().c_str());
}

void GLRenderer::step(int steps)
{
	steps = std::min(step_count + steps, max_frames) - step_count;

	update_time += scene->update(steps, std::chrono::duration<float>(timestep).count());

	step_count += steps;

	simulation_time += steps * timestep;

	if (step_count >= max_frames)
		reset();
}

void GLRenderer::reset()
{
	scene->reset();
	scene->update(1, 0.0f);
	last_update = std::chrono::steady_clock::now();
	simulation_time = std::chrono::nanoseconds(0);
	step_count = 0;
	update_time = 0.0f;
}

bool GLRenderer::toggle_freeze()
{
	frozen = !frozen;

	if (!frozen)
		last_update = std::chrono::steady_clock::now();

	return frozen;
}

bool GLRenderer::toggle_fps()
{
	return show_fps = !show_fps;
}

void GLRenderer::attach(GL::platform::MouseInputHandler* mouse_input)
{
	window.attach(mouse_input);
}

void GLRenderer::attach(GL::platform::KeyboardInputHandler* keyboard_input)
{
	window.attach(keyboard_input);
}

void GLRenderer::attach(GLScene* scene)
{
	this->scene = scene;
	reset();
}

image2D<std::uint32_t> GLRenderer::screenshot() const
{
	image2D<std::uint32_t> buffer(framebuffer_width, framebuffer_height);

	glReadBuffer(GL_FRONT);
	glReadPixels(0, 0, framebuffer_width, framebuffer_height, GL_RGBA, GL_UNSIGNED_BYTE, data(buffer));
	GL::throw_error();

	using std::swap;
	for (int y = 0; y < height(buffer) / 2; ++y)
		for (int x = 0; x < width(buffer); ++x)
			std::swap(buffer(x, y), buffer(x, height(buffer) - y - 1));

	return buffer;
}
