// Native Includes
#include <set>
#include <tuple>
#include <random>
#include <functional>

// Program Includes
#include "world.h"
#include "common.h"
#include "ga_gpu.h"

//Functions for both CPU and GPU
void make_world(World* world, int width, int height, int num_cities, int seed)
{
	// Random number generation
	mt19937::result_type rseed = seed;
	auto rgen = bind(uniform_real_distribution<>(0, 1), mt19937(rseed));
	init_world(world, width, height, num_cities);
	
	// Create a set to deal with uniqueness
	set<tuple<int, int>> coordinates;
	set<tuple<int, int>>::iterator it;
	pair<set<tuple<int, int>>::iterator,bool> ret;
	
	// Create some unique random cities
	for (int i=0; i<num_cities; i++)
	{
		while (true)
		{
			tuple<int,int> coors((int)(rgen() * width), (int)(rgen() * height));
			ret = coordinates.insert(coors);
			if (ret.second)
				break;
		}
	}
	
	// Add those cities to the world
	{
		int i = 0;
		for (it=coordinates.begin(); it!=coordinates.end(); it++)
		{
			world->cities[i].x = get<0>(*it);
			world->cities[i].y = get<1>(*it);
			i++;
		}
	}
}

//CPU

void init_world(World* world, int width, int height, int num_cities)
{
	world->width      = width;
	world->height     = height;
	world->num_cities = num_cities;
	world->fitness    = (float)0.0;
	world->fit_prob   = (float)0.0;
	world->cities     = new City[num_cities * sizeof(City)];
}

void clone_city(City* src, City* dst, int num_cities)
{
memcpy(dst, src, num_cities * sizeof(City));
}

void clone_world(World* src, World* dst)
{
	dst->width      = src->width;
	dst->height     = src->height;
	dst->num_cities = src->num_cities;
	dst->fitness    = src->fitness;
	dst->fit_prob   = src->fit_prob;
	clone_city(src->cities, dst->cities, src->num_cities);
}

void free_world(World* world)
{
	delete[] world->cities;
	delete[] world;
}

void free_population(World* pop, int pop_size)
{
	for (int i=0; i<pop_size; i++)
		delete[] pop[i].cities;
	delete[] pop;
}

//GPU

bool g_init_world(World* d_world, World* h_world)
{
	// Error checking
	bool error;
	
	// Soft clone world
	error = g_soft_clone_world(d_world, h_world);
	if (error)
		return true;
	
	// Allocate space for cities on device
	City *d_city;
	error = checkForError(cudaMalloc((void**)&d_city, h_world->num_cities * sizeof(City)));
	if (error)
	return true;
	
	// Update pointer on device
	error = checkForError(cudaMemcpy(&d_world->cities, &d_city, sizeof(City*), cudaMemcpyHostToDevice));
	if (error)
	return true;
	
	return false;
}

bool g_soft_clone_world(World* d_world, World* h_world)
{
	// Error checking
	bool error;
	
	error = checkForError(cudaMemcpy(&d_world->width, &h_world->width,        \
		sizeof(int), cudaMemcpyHostToDevice));
	if (error)
	return true;
	error = checkForError(cudaMemcpy(&d_world->height, &h_world->height,      \
		sizeof(int), cudaMemcpyHostToDevice));
	if (error)
	return true;
	error = checkForError(cudaMemcpy(&d_world->num_cities,                    \
		&h_world->num_cities, sizeof(int), cudaMemcpyHostToDevice));
	if (error)
	return true;

	return false;
}
