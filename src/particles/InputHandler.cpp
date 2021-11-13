#include <string_view>
#include <iostream>

#include <utils/screenshot.h>
#include <utils/io/png.h>

#include "GLScene.h"
#include "GLRenderer.h"
#include "InputHandler.h"

using namespace std::literals;


InputHandler::InputHandler(OrbitalNavigator& navigator, GLRenderer& renderer, GLScene& scene)
	: navigator(navigator), renderer(renderer), scene(scene)
{
	std::cout << R"""(controls:
	[space]             pause
	[right]             advance by one frame
	[page down]         advance by 10 frames
	[backspace]         reset
	[F2]                toggle bounding box
	[F8]                take screenshot

)"""sv << std::flush;
}

void InputHandler::keyDown(GL::platform::Key key, GL::platform::Window*)
{
	switch (key)
	{
	case GL::platform::Key::RIGHT:
		renderer.step();
		break;
	case GL::platform::Key::PAGE_DOWN:
		renderer.step(10);
		break;
	case GL::platform::Key::SPACE:
		renderer.toggle_freeze();
		break;
	case GL::platform::Key::BACKSPACE:
		renderer.reset();
		break;
	case GL::platform::Key::F1:
		renderer.toggle_fps();
		break;
	case GL::platform::Key::F2:
		scene.toggle_bounding_box();
		break;
	case GL::platform::Key::F8:
		PNG::saveImageR8G8B8A8(open_screenshot_file("Particles.png"), renderer.screenshot());
		break;
	default:
		break;
	}
}

void InputHandler::keyUp(GL::platform::Key, GL::platform::Window*)
{
}

void InputHandler::buttonDown(GL::platform::Button button, int x, int y, GL::platform::Window* window)
{
	navigator.buttonDown(button, x, y, window);
}

void InputHandler::buttonUp(GL::platform::Button button, int x, int y, GL::platform::Window* window)
{
	navigator.buttonUp(button, x, y, window);
}

void InputHandler::mouseMove(int x, int y, GL::platform::Window* window)
{
	navigator.mouseMove(x, y, window);
}

void InputHandler::mouseWheel(int d, GL::platform::Window* window)
{
	navigator.mouseWheel(d, window);
}
