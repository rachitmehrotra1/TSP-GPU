#include <iostream>
#include <algorithm>
#include "ga_cpu.h"
#include "common.h"
using namespace std;

void selection(World* pop, int pop_size, City** parents, float* rand_nums)
{

	// Select the parents
	for (int i=0; i<2; i++)
	{
		for (int j=0; j<pop_size; j++)
		{
			if (rand_nums[i] <= pop[j].fit_prob)
			{
				clone_city(pop[j].cities, parents[i], pop[0].num_cities *  sizeof(City));
				break;
			}
		}
	}
}

void crossover(City** parents, City* child, int num_cities, int cross_over)
{
	// Select elements in first parent from start up through crossover point
	clone_city(parents[0], child, cross_over + 1);
	
	// Add remaining elements from second parent to child, preserving order
	int remaining = num_cities - cross_over - 1;
	int count     = 0; 
	for (int i=0; i<num_cities; i++) // Loop parent
	{
		bool in_child = false;
		for (int j=0; j<=cross_over; j++) // Loop child
		{
			// If the city is in the child, exit this loop
			if ((child[j].x == parents[1][i].x) & (child[j].y == parents[1][i].y))
			{
				in_child = true;
				break;
			}
		}
			
		// If the city was not found in the child, add it to the child
		if (!in_child)
		{
			count++;
			clone_city(&parents[1][i], &child[cross_over + count], 1);
		}
		if (count == remaining) break;
	}
}

void mutate(City* child, int* rand_nums)
{
	City temp               = *(child + rand_nums[0]);
	*(child + rand_nums[0]) = *(child + rand_nums[1]);
	*(child + rand_nums[1]) = temp;
}

void execute(float prob_mutation, float prob_crossover, int pop_size,
	int max_gen, World* world, int seed)
{
	mt19937::result_type rseed = seed;
	auto rgen = bind(uniform_real_distribution<>(0, 1), mt19937(rseed));

	int world_size = pop_size * sizeof(World);
	World* old_pop = new World[world_size];
	World* new_pop = new World[world_size];
	int individual_size = world->num_cities * sizeof(City);

	// The best individuals
	int best_generation      = 0;
	World* best_leader       = new World[sizeof(World)];
	World* generation_leader = new World[sizeof(World)];
	init_world(best_leader, world->width, world->height, world->num_cities);
	init_world(generation_leader, world->width, world->height, world->num_cities);
	
	// Initialize the population
	initialize(world, old_pop, pop_size, seed);
	for (int i=0; i<pop_size; i++)
		init_world(&new_pop[i], world->width, world->height,world->num_cities);
	
	// Calculate the fitnesses
	float fit_sum = (float)0.0;
	for (int i=0; i<pop_size; i++)
	{
		old_pop[i].calc_fitness();
		fit_sum        += old_pop[i].fitness;
		old_pop[i].fit_prob = fit_sum;
	}
	// Compute the full probabilities
	for (int i=0; i<pop_size; i++)
		old_pop[i].fit_prob /= fit_sum;

	
	// Initialize the best leader
	select_leader(old_pop, pop_size, generation_leader, best_leader);
	print_status(generation_leader, best_leader, 0);


	// Continue through all generations
	for (int i=0; i<max_gen; i++)
	{

		// Create a new population
		for (int j=0; j<pop_size; j++)
		{
			// Parents and children
			City** parents = new City* [2];
			City* child    = new City[individual_size];
			parents[0]     = new City[individual_size];
			parents[1]     = new City[individual_size];
			
			// Generate all probabilities
			//Note : Same order as GPU to make sure same random values are generated
			float prob_select[2] = {(float)rgen(), (float)rgen()};
			float prob_cross     = (float)rgen();
			int   cross_loc      = (int)(rgen() * (world->num_cities - 1));
			float prob_mutate    = (float)rgen();
			int   mutate_loc[2]  = { (int)(rgen() * world->num_cities),(int)(rgen() * world->num_cities) };
			while (mutate_loc[1] == mutate_loc[0])
				mutate_loc[1] = (int)(rgen() * world->num_cities);

			// Select two parents
			selection(old_pop, pop_size, parents, &prob_select[0]);
			
			// Determine how many children are born
			if (prob_cross <= prob_crossover)
			{
				// Perform crossover
				crossover(parents, child, world->num_cities, cross_loc);

				// Perform mutation
				if (prob_mutate <= prob_mutation)
					mutate(child, &mutate_loc[0]);

				//Cild is added to new pop
				clone_city(child, new_pop[j].cities, individual_size);
			}
			else // Select the first parent
			{
				// Perform mutation
				if (prob_mutate <= prob_mutation)
					mutate(parents[0], &mutate_loc[0]);
				//Cild is added to new pop
				clone_city(parents[0], new_pop[j].cities, individual_size);
			}
		} 

		// Calculate the fitnesses
		float fit_sum = (float)0.0;
		for (int i=0; i<pop_size; i++)
		{
			new_pop[i].calc_fitness();
			fit_sum        += new_pop[i].fitness;
			new_pop[i].fit_prob = fit_sum;
		}
		// Compute the full probabilities
		for (int i=0; i<pop_size; i++)
			new_pop[i].fit_prob /= fit_sum;

		// Swap the populations
		World* temp = old_pop;
		old_pop     = new_pop;
		new_pop     = temp;

		// Select the new leaders
		if (select_leader(old_pop, pop_size, generation_leader, best_leader))
			best_generation = i + 1;
		print_status(generation_leader, best_leader, i + 1);
	}
	cout << endl << "Best generation found at " << best_generation << " generations" << endl;
	free_population(old_pop, pop_size);	free_population(new_pop, pop_size);
	free_world(best_leader); free_world(generation_leader);
}