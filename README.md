# Interacting Particles Simulation
In this document we will briefly describe the implementation of physical based particle collision system using a uniform grid structure. The main idea of uniform grid-based simulation for implementation for the project have been derived from the NVIDA white paper by S. Green [3]. The code in the repository contains our implementation of the same.

## Introduction
There are three main stages involved in the particle simulation.
1.	Particles setup
2.	Constructing a uniform grid structure
3.	Collision of large number of particles

The initial simulation involved containing the particles within a bounding box. At each position, the particle behaviour is described my Verlet integration version of Newtons equation of motions. Followed my inter-particles collision simulation by considering various types of forces acting on the particles during collisions. 

As a naïve collision algorithm of checking each particle with each other is not feasible option, we use a uniform spatial grid structure following by a simple position-based linear hashing to check for collisions. This algorithm is much faster than the naïve collision approach and is its implementation is within the code provided.

In the following sections, we first explain the uniform grid structure creation to facilitate localising the particles withing the gird. Next, we explain the hashing-based collision search algorithm to simulate particle collisions.

## Uniform Grid Structure

In this project, we follow uniform grid structure[1], we divide the entire space of the bounding box into a grid cells of equal size. Each particle could then be associated with a gird cell on which it overlaps.  As a direct benefit of this, the collision check between particles is much simpler task, given a specific cell, neighbouring cells are trivially located and the particles in those cells are checked for collision.

We use the two times the parameter `max_particle_radius` to set the grid cell resolution, we consider this parameter as the upper bound to the particle size. With a realistic assumption of no interparticle penetration and the particles can overlap with multiple grid cells, we resolve collisions within a cell by processing 27 neighbouring cells with pair wise tests. In our current implementation, the grid structure is generated from scratch at every time step and the performance is constant regardless of the particle position in the bounding box. 

In the next section we detail out the hashing-based grid structure generation and how we utilize it to perform collisions.

## Hashed Storage and Collisions
With a goal to use a dense array to store the entire gird structure, we map each grid cell into a hash table. We use a simple position-based linear hashing to wrap the particle coordinates into a hash table. In our implementation, the kernel `calculate_hash_kernel` calculates a hash value for each particle based on its bounding box coordinates. The result is stored in the global memory as `grid_hash_index` an `uint2` pair array (cell hash, particle id).

As the hash values are directly related to the position of the particles, we can determine the particles in the same grid cell and the neighbouring grid cells based on the hash values. Find the candidates for collision examination, we devise a two-step process:
1.	First, we sort the hash values of the particles using the Thrust library’s sorting method, a fast, efficient radix sorting method for CUDA-capable devices [2]. This provides us an array of particle ids ordered by grid cell positions.
2.	Next, we pass this array to the kernel `find_cell_start_kernel` to find the beginning particle index for given grid cell. The kernel compares the hash values of the current particle and previous particle from the sorted list. In those instances where the hash values do not match, indicates that the particle is in a different cell. We write this index value as the starting position for the new particle into the array `cell_start_idx` and ending position for the previous particle into the array `cell_end_idx`. In this manner we generate a starting and beginning position for all the cells.

![Figure 1](assets\report\UniformGrid_Sort.jpg)

The figure above shows the unifrom grid creation and assignment of cell starting indices on 4x4 grid world. Source:[3]

Finally, we utilize, the beginning and ending indices for each grid cells and the hash values of each particles to determine the forces acting on the particle. Based on the grid cell in which a particle is located, the method `calculate_acceleration` loops over 27 grid cells and checks for collisions for all the particles in those grid cells. If a collision is detected then, we update the particles position based on equations (1) – (9) [4].


## Conclusion
We want to bring it the notice to the evaluators of the project that this implementation could not be listed in the performance leader board as the implementation could not pass the benchmark evaluations. We suspect this could be due to mismatch in the CUDA compute capability mismatch. Unfortunately due to time constraints we could not fix the issues to make a successful submission.

In order to show the working of our implementation, we demonstrate the particle simulation on NVIDIA GTX 960 GPU (compute capability 3.0) and share the recording. The video can be found in the location: `assets\report\demo.mkv`

Google Drive Link: https://drive.google.com/file/d/1nhkxJII24xlIz3dTTiTMWe-yEGL4Mt8v/view?usp=sharing

![Final Result](assets\report\final.gif)

The gif above is a short clip of our final particles simulation. 

### References
[1] Ericson, C., Real-Time Collision Detection, Morgan Kaufmann 2005

[2] https://docs.nvidia.com/cuda/thrust/index.html#sorting

[3] Simon Green. 2010. Particle simulation using cuda. NVIDIA Whitepaper 6 (2010), 121–128.

[4] Assignment 4b, GPU Programming Course, Winter Semester 2020, Saarland University.

