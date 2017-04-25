#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <curand_kernel.h>

//Problem Parameters
#define CITIES 100
#define ANTS 2000
#define MAX_DIST 100
#define ALPHA 1
#define BETA 5       //This parameter raises the weight of distance over pheromone
#define RHO 0.5      //Evapouration rate
#define QVAL 100     //
#define MAX_TOURS 50// The number of times an ant will walk trough all the cities
#define INIT_PHER (1.0/CITIES) //Initial hormone for each path
#define MAX_TOTAL_DISTANCE (CITIES * MAX_DIST) // MAX possible distance that an ant can walk


struct ant{
  int curCity, nextCity, pathIndex;
  int visited[CITIES];
  int path[CITIES];
  float tourLength;
};

//CPU
float        distances[CITIES][CITIES]; // Distance between city i an j
double       hormone[CITIES][CITIES]; //Hormone between city i and j
struct ant   ants[ANTS];
float        bestdistance[ANTS];
float 		 finalbest = (float)MAX_TOTAL_DISTANCE;
curandState  state[ANTS];
const size_t distances_size = sizeof(float) * size_t(CITIES*CITIES);
const size_t hormone_size = sizeof(double) * size_t(CITIES*CITIES);
//GPU
float        *distances_d;
struct ant   *ants_d;
double       *hormone_d;
float        *bestdistance_d;
curandState  *state_d;
int BLOCKS,THREADS;

void get_distances_matrix();
void deviceAlloc();
__global__ void initialize_ants(struct ant *ants_d, curandState *state_d, float *bestdistance_d , int THREADS);
__global__ void setup_curand_states(curandState *state_d, unsigned long t , int THREADS);
__global__ void restart_ants(struct ant *ants_d,curandState *state_d, float *bestdistance_d , int THREADS);
void move_ants();
__global__ void simulate_ants(struct ant *ants_d,curandState *state_d, float *distances_d, double *hormone_d, int THREADS);
__device__ double antProduct(int from, int to, double *hormone_d, float *distances_d);
__device__ int NextCity(struct ant *ants_d, int pos, float *distances_d, double *hormone_d, curandState *state_d );
void updateTrails();

int main(){

		//Set blocks and threads based on number of ants
		if(ANTS<=1024)
		{
			BLOCKS=1;
			THREADS=ANTS;
		}
		else
		{
			THREADS=1024;
			BLOCKS=ceil(ANTS/(float)THREADS);

		}
		get_distances_matrix(); // Get the distances between cities from the input
		deviceAlloc(); // Mallocs and memcpy of the device variables

		//Set up an array of curand_states in order to build better random numbers
		time_t t; time(&t);
		setup_curand_states <<< BLOCKS, THREADS >>> (state_d, (unsigned long) t , THREADS);
		cudaThreadSynchronize();

		//initialize the ants array
		initialize_ants <<< BLOCKS, THREADS >>> (ants_d, state_d, bestdistance_d , THREADS);
		cudaThreadSynchronize();

		// Start and control the ants tours
		move_ants();

		//Free Memory
		cudaFree(ants_d);
		cudaFree(bestdistance_d);
		cudaFree(distances_d);
		cudaFree(hormone_d);
		cudaFree(state_d);
		cudaFree(bestdistance_d);

		return 0;
}


void get_distances_matrix(){
  int i,j;
  float k;

  while(scanf("%i %i %f", &i,&j,&k) == 3){
    distances[i][j] = k;
    hormone[i][j] = INIT_PHER;
  }

}

void deviceAlloc(){
	cudaMalloc( (void**) &ants_d, sizeof(ants));
	cudaMalloc( (void**) &state_d, sizeof(state));

	cudaMalloc( (void**) &distances_d, distances_size);
	cudaMemcpy(distances_d, distances, distances_size, cudaMemcpyHostToDevice);

	cudaMalloc( (void**) &hormone_d, hormone_size);
	cudaMemcpy(hormone_d, hormone, hormone_size, cudaMemcpyHostToDevice);

	cudaMalloc( (void**) &bestdistance_d, sizeof(bestdistance));
}

__global__ void setup_curand_states(curandState *state_d, unsigned long t, int THREADS){
	int id = threadIdx.x + blockIdx.x*THREADS;
	curand_init(t, id, 0, &state_d[id]);
}

__global__ void initialize_ants(struct ant *ants_d, curandState *state_d, float *bestdistance_d , int THREADS){

  int position = threadIdx.x + blockIdx.x*THREADS;
  int k;

  // Mark all cities as not visited
  // Mark all path as not traversed
  for(k = 0; k < CITIES; k++){
    ants_d[position].visited[k] = 0;
    ants_d[position].path[k] = -1;
  }

  bestdistance_d[position] = (float)MAX_TOTAL_DISTANCE;

  //Random City to begin
  ants_d[position].curCity = curand(&state_d[position])% CITIES;
  ants_d[position].pathIndex = 1;
  ants_d[position].path[0] = ants_d[position].curCity;
  ants_d[position].nextCity = -1;
  ants_d[position].tourLength = 0;
  ants_d[position].visited[ants_d[position].curCity] = 1;
}

__global__ void restart_ants(struct ant *ants_d,curandState *state_d, float *bestdistance_d , int THREADS){

	int position = threadIdx.x + blockIdx.x*THREADS;
	int i;

	if(ants_d[position].tourLength < bestdistance_d[position]){
		bestdistance_d[position] = ants_d[position].tourLength;
	}

	ants_d[position].nextCity = -1;
	ants_d[position].tourLength = 0.0;

	for(i = 0; i < CITIES; i++){
		ants_d[position].visited[i] = 0;
		ants_d[position].path[i] = -1;
	}

	ants_d[position].curCity = curand(&state_d[position])% CITIES;
	ants_d[position].pathIndex = 1;
	ants_d[position].path[0] = ants_d[position].curCity;
	ants_d[position].visited[ants_d[position].curCity] = 1;
}

void move_ants(){
	int curtour = 0;
	while (curtour++ < MAX_TOURS){
		simulate_ants <<< BLOCKS, THREADS >>> (ants_d, state_d, distances_d, hormone_d, THREADS);
		cudaThreadSynchronize();

		cudaMemcpy(ants, ants_d, sizeof(ants), cudaMemcpyDeviceToHost);
		
		//update the trails of the ants
					int from,to,i,ant;

				//hormone evaporation
				for(from = 0; from < CITIES; from++)
					for(to = 0;to < CITIES; to++){
						if(from!=to){
							hormone[from][to] *=( 1.0 - RHO);

							if(hormone[from][to] < 0.0){
								hormone[from][to] = INIT_PHER;
							}
						}
					}

				//add new pheromone to the trails
				for(ant = 0; ant < ANTS; ant++)
					for(i = 0; i < CITIES; i++){
						if( i < CITIES - 1 ){
							from = ants[ant].path[i];
							to = ants[ant].path[i+1];
						}
						else{
							from = ants[ant].path[i];
							to = ants[ant].path[0];
						}

						hormone[from][to] += (QVAL/ ants[ant].tourLength);
						hormone[to][from] = hormone[from][to];

					}


				for (from = 0; from < CITIES; from++)
					for( to = 0; to < CITIES; to++){
						hormone[from][to] *= RHO;
					}

		cudaMemcpy(hormone_d, hormone, hormone_size, cudaMemcpyHostToDevice);
		cudaMemcpy(bestdistance, bestdistance_d, sizeof(bestdistance), cudaMemcpyDeviceToHost);
				for(i =0; i < ANTS; i++)
				  if(bestdistance[i] < finalbest){
					  finalbest = bestdistance[i];
				  }
				printf("Best distance %f \n", finalbest);

		restart_ants <<< BLOCKS, THREADS >>> (ants_d, state_d, bestdistance_d, THREADS);
		cudaThreadSynchronize();

	}
}

__global__ void simulate_ants(struct ant *ants_d,curandState *state_d, float *distances_d, 
								double *hormone_d , int THREADS ){

	int position = threadIdx.x + blockIdx.x*THREADS;
	int curtime = 0;

	while(curtime++ < CITIES){
		//check if all cities were visited
		if( ants_d[position].pathIndex < CITIES ){ 

			ants_d[position].nextCity = NextCity(ants_d, position, distances_d, hormone_d, state_d);
			ants_d[position].visited[ants_d[position].nextCity] = 1;
			ants_d[position].path[ants_d[position].pathIndex++] = ants_d[position].nextCity;
			ants_d[position].tourLength += distances_d[ants_d[position].curCity + (ants_d[position].nextCity * CITIES)];
			if(ants_d[position].pathIndex == CITIES){
				ants_d[position].tourLength += distances_d[ants_d[position].path[CITIES -1] + (ants_d[position].path[0]*CITIES)];
			}
			ants_d[position].curCity = ants_d[position].nextCity;
		}
	}

}

__device__ double antProduct(int from, int to, double *hormone_d, float *distances_d){
  return (double) (( pow( hormone_d[from + to*CITIES], ALPHA) * pow( (1.0/ distances_d[from + to*CITIES]), BETA)));
}


__device__ int NextCity(struct ant *ants_d, int pos, float *distances_d, double *hormone_d, curandState *state_d ){
	int to, from;
	double denom = 0.0;
	from =  ants_d[pos].curCity;

	for(to = 0; to < CITIES; to++){
	  if(ants_d[pos].visited[to] == 0){
		denom += antProduct(from, to, hormone_d, distances_d);
	  }
	}

	assert(denom != 0.0);

	to++;
	int count = CITIES - ants_d[pos].pathIndex;

	do{
		double p;
		to++;

		if(to >= CITIES)
			to = 0;

		if(ants_d[pos].visited[to] == 0){
			p = (double) antProduct(from, to, hormone_d, distances_d)/denom;
			double x = (double)(curand(&state_d[pos])% 1000000000000000000)/1000000000000000000;
			if(x < p){
				break;
			}
			count--;
			if(count == 0){
				break;
			}
		}
	}while(1);

	return to;
}

