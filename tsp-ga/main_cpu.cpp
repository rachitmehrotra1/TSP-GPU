#include <ctime>
#include <iostream>
#include <string.h>
#include <sstream>
#include <cstring>
#include <string>
#include <cstdio>
#include "common.h"
#include "ga_cpu.h"

using namespace std;

int main()
{
	// GA parameters
	float prob_mutation  = (float)0.15; // The probability of a mutation
	float prob_crossover = (float)0.8;  // The probability of a crossover
	int world_seed       = 12438955;    // Seed for initial city selection
	int ga_seed          = 87651111;    // Seed for all other random numbers
	
	// World parameters
	int world_width  = 10000;
	int world_height = 10000;
	
	// The test cases
	int iterations          = 1;  // Number of full runs
	const int num_cases     = 1; // How many trials to test
	int cases[num_cases][3] =     // num_cities, pop_size, max_gen
	{
		// {25, 100,    1000},
		// {25, 1000,   1000},
		// {25, 10000,  100},
		// {25, 100000, 10},

		// {50, 100,    1000},
		// {50, 1000,   1000},
		// {50, 10000,  100},
		// {50, 100000, 10},

		// {100, 100,    1000},
		// {100, 1000,   1000},
		// {100, 10000,  100},
		{100, 100000, 10}

	};

	for (int i=0; i<num_cases; i++)
	{
		// GA params
		int num_cities = cases[i][0];
		int pop_size   = cases[i][1];
		int max_gen    = cases[i][2];

		// Make the world
		World* world = new World[sizeof(World)];
		make_world(world, world_width, world_height, num_cities, world_seed);
		cout << "CPU Version - START" << endl;
		for (int j=0; j<iterations; j++)
		{
			execute(prob_mutation, prob_crossover, pop_size, max_gen, world, ga_seed);
		}

		cout << "CPU - END" << endl;

		free_world(world);
	}
	
	return 0;
}
