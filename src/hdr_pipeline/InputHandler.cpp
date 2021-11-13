#include <utils/io/png.h>
#include <utils/screenshot.h>

#include "GLRenderer.h"
#include "InputHandler.h"


InputHandler::InputHandler(OrbitalNavigator& navigator, GLRenderer& renderer)
	: navigator(navigator), renderer(renderer)
{
}

void InputHandler::keyDown(GL::platform::Key key, GL::platform::Window*)
{
	switch (key)
	{
	case GL::platform::Key::F8:
		PNG::saveImageR8G8B8A8(open_screenshot_file("HDRPipeline.png"), renderer.screenshot());
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
