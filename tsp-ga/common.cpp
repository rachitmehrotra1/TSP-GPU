// Native includes
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <algorithm>

// Program includes
#include "common.h"

using namespace std;

bool checkForError(cudaError_t error)
{
	if (error != cudaSuccess)
	{
		cout << cudaGetErrorString(error) << endl;
		return true;
	}
	else
	{
		return false;
	}
}
void initialize(World* world, World* pop, int pop_size, int seed)
{
	// Set the seed for random number generation
	srand(seed);

	for (int i=0; i<pop_size; i++)
	{
		// Clone world
		pop[i].cities = new City[world->num_cities * sizeof(City)];
		clone_world(world, &pop[i]);

		// Randomly adjust the path between cities
		random_shuffle(&pop[i].cities[0], &pop[i].cities[world->num_cities]);
	}
}

int select_leader(World* pop, int pop_size, World* generation_leader,
	World* best_leader)
{
	// Find element with the largest fitness function
	int ix = 0;
	for (int i=1; i<pop_size; i++)
	{
		if (pop[i].fitness > pop[ix].fitness)
			ix = i;
	}

	// Store generation leader
	clone_world(&pop[ix], generation_leader);

	// Update best leader
	if (generation_leader->fitness > best_leader->fitness)
	{
		clone_world(generation_leader, best_leader);
		return 1;
	}

	return 0;
}

void print_status(World* generation_leader, World* best_leader, int generation)
{
	cout << "Generation " << generation << ":" << endl;
	cout << "  Current Leader's Fitness: "  << generation_leader->fitness << endl;
	cout << "  Best Leader's Fitness: "  << best_leader->fitness << endl;
}