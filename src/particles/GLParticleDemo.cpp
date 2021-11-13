#include <iostream>

#include <GL/platform/Application.h>

#include <utils/PerspectiveCamera.h>
#include <utils/OrbitalNavigator.h>
#include <utils/image.h>

#include "InputHandler.h"
#include "GLRenderer.h"
#include "GLParticleDemo.h"
#include "ParticleDemo.h"


void ParticleDemo::run(std::filesystem::path output_file, const std::filesystem::path& input_file, int N, int subsample, float dt, int cuda_device)
{
	if (!output_file.empty())
		std::cerr << "WARNING: output file ignored in interactive mode\n";

	GLRenderer renderer(N, dt, false);

	auto [scene, bb_min, bb_max] = load_scene(input_file, cuda_device);

	PerspectiveCamera camera(60.0f * math::pi<float> / 180.0f, 0.1f, length(bb_max - bb_min) * 10.0f);
	OrbitalNavigator navigator(-math::pi<float> / 2, 0.0f, length(bb_max - bb_min) * 1.5f, (bb_min + bb_max) * 0.5f);

	InputHandler input_handler(navigator, renderer, *scene);

	renderer.attach(static_cast<GL::platform::MouseInputHandler*>(&input_handler));
	renderer.attach(static_cast<GL::platform::KeyboardInputHandler*>(&input_handler));
	renderer.attach(scene.get());

	camera.attach(&navigator);
	scene->attach(&camera);

	GL::platform::run(renderer);
}
