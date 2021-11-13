#include <utility>
#include <stdexcept>
#include <variant>
#include <vector>
#include <iostream>

#include <utils/CUDA/error.h>
#include <utils/CUDA/device.h>

#include "particle_system_state.h"

#include "ParticleSystemLoader.h"
#include "GLCUDAParticles.h"
#include "GLParticleReplay.h"
#include "GLParticleDemo.h"


std::tuple<std::unique_ptr<GLScene>, math::float3, math::float3> load_scene(const std::filesystem::path& path, int cuda_device)
{
	CUDA::print_device_properties(std::cout, cuda_device) << '\n' << '\n' << std::flush;
	throw_error(cudaSetDevice(cuda_device));

	class SceneBuilder : private virtual ParticleSystemBuilder
	{
		struct simulation_data
		{
			std::size_t num_particles;
			std::unique_ptr<float[]> position;
			std::unique_ptr<std::uint32_t[]> color;
			ParticleSystemParameters params;
		};

		std::variant<simulation_data, GLParticleReplayBuilder> data;


		void add_particle_simulation(std::size_t num_particles, std::unique_ptr<float[]> position, std::unique_ptr<std::uint32_t[]> color, const ParticleSystemParameters& params) override
		{
			data.emplace<simulation_data>(simulation_data { num_particles, std::move(position), std::move(color), params });
		}

		ParticleReplayBuilder& add_particle_replay(std::size_t num_particles, const float* position, const std::uint32_t* color, const ParticleSystemParameters& params) override
		{
			return data.emplace<GLParticleReplayBuilder>(num_particles, position, color, params);
		}

	public:
		auto load(const std::filesystem::path& path)
		{
			load_particles(*this, path);

			struct ParticleSystemFactory
			{
				std::tuple<std::unique_ptr<GLScene>, math::float3, math::float3> operator ()(simulation_data& data)
				{
					static auto module = particle_system_module("particle_system");

					math::float3 bb_min = { data.params.bb_min[0], data.params.bb_min[1], data.params.bb_min[2] };
					math::float3 bb_max = { data.params.bb_max[0], data.params.bb_max[1], data.params.bb_max[2] };

					auto particles = module.create_instance(data.num_particles, &data.position[0] + 0 * data.num_particles, &data.position[0] + 1 * data.num_particles, &data.position[0] + 2 * data.num_particles, &data.position[0] + 3 * data.num_particles, &data.color[0], data.params);

					return {
						std::make_unique<GLCUDAParticles>(std::move(particles), data.num_particles, std::move(data.position), std::move(data.color), bb_min, bb_max),
						bb_min,
						bb_max
					};
				}

				auto operator ()(GLParticleReplayBuilder& builder)
				{
					return builder.finish();
				}
			};

			return std::visit(ParticleSystemFactory(), data);
		}
	};

	SceneBuilder scene_builder;
	return scene_builder.load(path);
}
