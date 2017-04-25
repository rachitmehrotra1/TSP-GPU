#ifndef __GA_CPU_H__
#define __GA_CPU_H__

// Program includes
#include "world.h"


void selection(World* pop, int pop_size, City** parents, float* rand_nums);
void crossover(City** parents, City* child, int num_cities, int cross_over);
void mutate(City* child, int* rand_nums);
void execute(float prob_mutation, float prob_crossover, int pop_size,
	int max_gen, World* world, int seed);

#endif