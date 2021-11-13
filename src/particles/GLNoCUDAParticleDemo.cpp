#include <utility>
#include <stdexcept>
#include <optional>
#include <vector>
#include <iostream>

#include "particle_system_state.h"

#include "ParticleSystemLoader.h"
#include "GLParticleReplay.h"
#include "GLParticleDemo.h"


std::tuple<std::unique_ptr<GLScene>, math::float3, math::float3> load_scene(const std::filesystem::path& path, int cuda_device)
{
	class SceneBuilder : private virtual ParticleSystemBuilder
	{
		std::optional<GLParticleReplayBuilder> builder;


		void add_particle_simulation(std::size_t num_particles, std::unique_ptr<float[]> position, std::unique_ptr<std::uint32_t[]> color, const ParticleSystemParameters& params) override
		{
			throw std::runtime_error("cannot run particle system without CUDA");
		}

		ParticleReplayBuilder& add_particle_replay(std::size_t num_particles, const float* position, const std::uint32_t* color, const ParticleSystemParameters& params) override
		{
			return builder.emplace(num_particles, position, color, params);
		}

	public:
		std::tuple<std::unique_ptr<GLScene>, math::float3, math::float3> load(const std::filesystem::path& path)
		{
			load_particles(*this, path);
			return builder->finish();
		}
	};

	SceneBuilder scene_builder;
	return scene_builder.load(path);
}
