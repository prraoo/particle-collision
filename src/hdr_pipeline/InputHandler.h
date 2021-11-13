#ifndef INCLUDED_INPUT_HANDLER
#define INCLUDED_INPUT_HANDLER

#pragma once

#include <GL/platform/InputHandler.h>

#include <utils/OrbitalNavigator.h>


class GLRenderer;

class InputHandler : public virtual GL::platform::KeyboardInputHandler, public virtual GL::platform::MouseInputHandler
{
	OrbitalNavigator& navigator;
	GLRenderer& renderer;

public:
	InputHandler(OrbitalNavigator& navigator, GLRenderer& renderer);

	InputHandler(const InputHandler&) = delete;
	InputHandler& operator =(const InputHandler&) = delete;


	void keyDown(GL::platform::Key key, GL::platform::Window*) override;
	void keyUp(GL::platform::Key key, GL::platform::Window*) override;
	void buttonDown(GL::platform::Button button, int x, int y, GL::platform::Window*) override;
	void buttonUp(GL::platform::Button button, int x, int y, GL::platform::Window*) override;

	void mouseMove(int x, int y, GL::platform::Window*) override;
	void mouseWheel(int delta, GL::platform::Window*) override;
};

#endif  // INCLUDED_INPUT_HANDLER
