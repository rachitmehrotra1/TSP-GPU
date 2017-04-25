#ifndef __WORLD_H__
#define __WORLD_H__

// Native Includes
#include <iostream>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstring>
using namespace std;
struct City
{
//Store location - city
	int x, y;
};

typedef struct World
{
	// 2D world for the TSP
	int width, height; 
	int num_cities;    
	City* cities;      
	float fitness;     // The current fitness
	float fit_prob;    // The fitness probability

	inline __host__ void calc_fitness()
	{
		float distance = 0.0;
		for (int i=0; i<num_cities-1; i++)
			distance += (cities[i].x - cities[i + 1].x) * (cities[i].x -      \
				cities[i +1 ].x) + (cities[i].y - cities[i + 1].y)     *      \
				(cities[i].y - cities[i + 1].y);
		fitness = (width * height) / distance;
	}

	inline __host__ float calc_distance()
	{
		float distance = 0.0;
		for (int i=0; i<num_cities-1; i++)
			distance += (float)sqrt((float)((cities[i].x - cities[i + 1].x) * \
				(cities[i].x - cities[i + 1].x) + (cities[i].y              - \
				cities[i +1 ].y) * (cities[i].y - cities[i + 1].y)));
		return distance;
	}
} World;


// Makes a new world struct
void make_world(World* world, int width, int height, int num_cities, int seed);


//FOR CPU


void init_world(World* world, int width, int height, int num_cities);
void clone_city(City* src, City* dst, int num_cities);
void clone_world(World* src, World* dst);
void free_world(World* world);
void free_population(World* pop, int pop_size);

//FOR GPU 
//d_* means device 
//h_* means host
bool g_init_world(World* d_world, World* h_world);
bool g_soft_clone_world(World* d_world, World* h_world);

#endif
