#ifndef __COMMON_H__
#define __COMMON_H__

#include <ctime>
#include <random>
#include "world.h"

 
// Check for a CUDA error
bool checkForError(cudaError_t error);
// Initialize the population in host memory
void initialize(World* world, World* pop, int pop_size, int seed);
// Updates the generation and global best leaders	
int select_leader(World* pop, int pop_size, World* generation_leader,
	World* best_leader);
void print_status(World* generation_leader, World* best_leader, \
	int generation);

#endif