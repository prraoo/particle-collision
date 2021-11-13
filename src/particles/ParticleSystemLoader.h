#ifndef INCLUDED_PARTICLE_SYSTEM_LOADER
#define INCLUDED_PARTICLE_SYSTEM_LOADER

#pragma once

#include <memory>

#include <utils/dynamic_library.h>

#include "particle_system_module.h"


struct particle_system_deleter
{
	destroy_particles_func* destroy;

	void operator ()(ParticleSystem* particles) const
	{
		destroy(particles);
	}
};

using unique_particle_system = std::unique_ptr<ParticleSystem, particle_system_deleter>;


class particle_system_instance
{
	update_particles_func* update_func;
	reset_particles_func* reset_func;
	unique_particle_system particles;

public:
	particle_system_instance(update_particles_func* update_func, reset_particles_func* reset_func, unique_particle_system particles)
		: update_func(update_func), reset_func(reset_func), particles(std::move(particles))
	{
	}

	void reset(const float* x, const float* y, const float* z, const float* r, const std::uint32_t* color)
	{
		reset_func(particles.get(), x, y, z, r, color);
	}

	void update(float* position, std::uint32_t* color, float dt)
	{
		update_func(particles.get(), position, color, dt);
	}
};

class particle_system_module
{
	unique_library module;

	create_particles_func* create_particles;
	reset_particles_func* reset_particles;
	update_particles_func* update_particles;
	destroy_particles_func* destroy_particles;

public:
	particle_system_module(const std::filesystem::path& path)
		: module(load_library(path)),
		  create_particles(lookup_symbol<create_particles_func>(module.get(), "create_particles")),
		  reset_particles(lookup_symbol<reset_particles_func>(module.get(), "reset_particles")),
		  update_particles(lookup_symbol<update_particles_func>(module.get(), "update_particles")),
		  destroy_particles(lookup_symbol<destroy_particles_func>(module.get(), "destroy_particles"))
	{
	}

	auto create_instance(std::size_t num_particles, const float* x, const float* y, const float* z, const float* r, const std::uint32_t* color, const ParticleSystemParameters& params)
	{
		return particle_system_instance(update_particles, reset_particles, unique_particle_system(create_particles(num_particles, x, y, z, r, color, params), { destroy_particles }));
	}
};

#endif  // INCLUDED_PARTICLE_SYSTEM_LOADER
