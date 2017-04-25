#ifndef __GA_GPU_H__
#define __GA_GPU_H__

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "world.h"


bool checkForKernelError(const char *err_msg);
__device__ int getGlobalIdx_2D_2D();

// Perform crossover on the device
// sel_ix    : The indexes of the parents in the old population
// cross_loc : The crossover locations
__device__ void crossover(World* old_pop, World* new_pop, int* sel_ix,int* cross_loc, int tid);
__device__ void mutate(World* new_pop, int* mutate_loc, int tid);
__global__ void fitness_kernel(World* pop, int pop_size);
__global__ void fit_sum_kernel(World* pop, int pop_size, float* fit_sum);
__global__ void fit_prob_kernel(World* pop, int pop_size, float* fit_sum);
__global__ void max_fit_kernel(World* pop, int pop_size, float* max, int* ix);
__global__ void selection_kernel(World* pop, int pop_size, float* rand_nums, int* sel_ix);


// Main kernal for creating children of the population
// 	old_pop        : The old population (where the parents are located)
// 	new_pop        : The new population (where the children will be)
// 	pop_size       : The number of elements in the population
// 	sel_ix         : The indexes of the parents in the old population
// 	prob_crossover : The probability of corssover occuring
// 	prob_cross     : The probabilities of crossover occuring
// 	cross_loc      : The crossover locations
// 	prob_mutation  : The probability of mutation occuring
// 	prob_mutate    : The probabilities of mutation occuring
// 	mutate_loc     : The mutation locations
__global__ void child_kernel(World* old_pop, World* new_pop, int pop_size, 
	int* sel_ix, float prob_crossover, float* prob_cross, int* cross_loc, 
	float prob_mutation, float* prob_mutate, int* mutate_loc);

bool g_initialize(World* world, World* pop, int pop_size, int seed);

bool g_evaluate(World *pop, int pop_size, dim3 Block, dim3 Grid);

int g_select_leader(World* pop, int pop_size, World* generation_leader,World* best_leader, dim3 Block, dim3 Grid);

bool g_execute(float prob_mutation, float prob_crossover, int pop_size,int max_gen, World* world, int seed);

#endif