#ifndef INCLUDED_GL_PARTICLE_DEMO
#define INCLUDED_GL_PARTICLE_DEMO

#include <memory>
#include <tuple>
#include <filesystem>

#include <utils/math/vector.h>

#include "GLScene.h"


std::tuple<std::unique_ptr<GLScene>, math::float3, math::float3> load_scene(const std::filesystem::path& path, int cuda_device);

#endif  // INCLUDED_GL_PARTICLE_DEMO
