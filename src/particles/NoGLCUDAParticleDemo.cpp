#include <utility>
#include <iostream>
#include <iomanip>
#include <filesystem>

#include <utils/dynamic_library.h>

#include <utils/CUDA/error.h>
#include <utils/CUDA/device.h>

#include <utils/math/vector.h>

#include "particle_system_state.h"
#include "ParticleSystemLoader.h"
#include "CUDAParticles.h"
#include "ParticleDemo.h"


namespace
{
	auto load(std::ostream& output_file, const std::filesystem::path& path)
	{
		class SceneBuilder : private virtual ParticleSystemBuilder, private virtual ParticleReplayBuilder
		{
			std::size_t num_particles;
			std::unique_ptr<float[]> position;
			std::unique_ptr<std::uint32_t[]> color;
			ParticleSystemParameters params;

			void add_particle_simulation(std::size_t num_particles, std::unique_ptr<float[]> position, std::unique_ptr<std::uint32_t[]> color, const ParticleSystemParameters& params) override
			{
				this->num_particles = num_particles;
				this->position = std::move(position);
				this->color = std::move(color);
				this->params = params;
			}

			ParticleReplayBuilder& add_particle_replay(std::size_t num_particles, const float* position, const std::uint32_t* color, const ParticleSystemParameters& params) override
			{
				throw std::runtime_error("particle replay not supported in non-interactive mode");
				return *this;
			}

			void add_frame(std::chrono::nanoseconds dt, const float* positions, const std::uint32_t* colors) override
			{
			}

		public:
			auto load(std::ostream& output_file, const std::filesystem::path& path)
			{
				load_particles(*this, path);

				if (!position)
					throw std::runtime_error("file did not contain a particle system");

				static auto module = particle_system_module("particle_system");

				struct result_t { ParticleReplayWriter writer; CUDAParticles particles; };

				return result_t {
					ParticleReplayWriter(output_file, num_particles, &position[0] + 0 * num_particles, &position[0] + 1 * num_particles, &position[0] + 2 * num_particles, &position[0] + 3 * num_particles, &color[0], params),
					CUDAParticles(module, num_particles, std::move(position), std::move(color), params)
				};
			}
		};

		SceneBuilder scene_builder;
		return scene_builder.load(output_file, path);
	}

	std::ostream& pad(std::ostream& out, int n)
	{
		for (int i = n; i > 0; --i) out.put(' ');
		return out;
	}
}

void ParticleDemo::run(std::filesystem::path output_file, const std::filesystem::path& input_file, int N, int subsample, float dt, int cuda_device)
{
	CUDA::print_device_properties(std::cout, cuda_device) << '\n' << '\n' << std::flush;
	throw_error(cudaSetDevice(cuda_device));

	if (output_file.empty())
		output_file = std::filesystem::path(input_file).filename().replace_extension(".particlereplay");

	{
		std::ofstream out(output_file, std::ios::binary);

		if (!out)
			throw std::runtime_error("failed to open output file \"" + output_file.string() + '"');

		auto [writer, particles] = load(out, input_file);


		float particles_time = 0.0f;

		std::cout << '\n' << N << " frame(s):\n";

		int padding = static_cast<int>(std::log10(N));
		int next_padding_shift = 10;

		for (int i = 0; i < N; i += subsample)
		{
			auto t = particles.update(out, writer, subsample, dt) / subsample;

			if ((i + 1) >= next_padding_shift)
			{
				--padding;
				next_padding_shift *= 10;
			}

			pad(std::cout, padding) << "t_" << (i + 1) << ": " << std::setprecision(2) << std::fixed << t << " ms\n" << std::flush;

			particles_time += t;
		}

		std::cout << "avg time: " << std::setprecision(2) << std::fixed << particles_time / N << " ms\n" << std::flush;

		writer.finish(out);
	}
}
