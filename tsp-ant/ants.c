#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

//Problem parameters
#define CITIES 100
#define ANTS 2000
#define MAX_DIST 100
#define MAX_TOTAL_DISTANCE (CITIES * MAX_DIST)

#define ALPHA 1
#define BETA 5 //This parameter raises the weight of distance over pheromone
#define RHO 0.5 //Evapouration rate
#define QVAL 100
#define MAX_TOURS 50
#define MAX_TIME (MAX_TOURS * CITIES)
#define INIT_PHER (1.0/CITIES)

//Global structures
struct ant{
	
	int curCity, nextCity, pathIndex;
	int visited[CITIES];
	int path[CITIES];
	float tourLength;
};

float distances[CITIES][CITIES];
struct ant ants[ANTS];
double hormone[CITIES][CITIES];

float bestdistance = (float)MAX_TOTAL_DISTANCE;
int bestIndex;

//Methods

void get_distances_matrix(){
  int i= 0,j = 0;
  double k;

  while(scanf("%i %i %lf",&i,&j,&k) == 3){
    distances[i][j] = k;
    hormone[i][j] = INIT_PHER;
     // printf("i: %i, j: %i, k: %lf \n",i,j,k);
     // printf("Distance[i][j]: %lf\n", distances[i][j]);
  }
 printf("Got distance Matrix -- %i cities\n", CITIES);
}

void initialize_ants(){
   int i,k, init = 0;
   for( i = 0; i < ANTS; i++){
     if(init == CITIES)
       init = 0;
  
     for(k = 0; k < CITIES; k++){
       ants[i].visited[k] = 0;
       ants[i].path[k] = -1;
     } 

     ants[i].curCity = init++;
     ants[i].pathIndex = 1;
     ants[i].path[0] = ants[i].curCity;
     ants[i].nextCity = -1;
     ants[i].tourLength = 0;
     ants[i].visited[ants[i].curCity] = 1;
   }
  // printf("Ants Initialized - %i ants\n",ANTS);
}

//reinitialize all ants and redistribute them
void restart_ants(){
	int ant,i,to=0;

	for(ant = 0; ant < ANTS; ant++){
	  // printf("ant %i -- tour: %f  -- best: %f\n",ant, ants[ant].tourLength, bestdistance);
		if(ants[ant].tourLength < bestdistance){
			bestdistance = ants[ant].tourLength;
			bestIndex = ant;
		}

		ants[ant].nextCity = -1;
		ants[ant].tourLength = 0.0;

		for(i = 0; i < CITIES; i++){
			ants[ant].visited[i] = 0;
			ants[ant].path[i] = -1;
		}
		
		ants[ant].curCity = rand()%CITIES;

		ants[ant].pathIndex = 1;
		ants[ant].path[0] = ants[ant].curCity;

		ants[ant].visited[ants[ant].curCity] = 1;
	}
}

double antProduct(int from, int to){
  // printf("Ant Product: from: %i to: %i\n", from, to);
  // printf("First: %lf, Second: %lf\n",pow( hormone[from][to], ALPHA), pow( (1.0/ distances[from][to]), BETA));
  // printf("Hormone: %lf, Distance: %lf, BETA: %lf\n", hormone[from][to], distances[from][to], BETA);
   return (double) (( pow( hormone[from][to], ALPHA) * pow( (1.0/ distances[from][to]), BETA)));
}

int NextCity( int pos ){
	int from, to;
	double denom = 0.0;

	from = ants[pos].curCity;

  for(to = 0; to < CITIES; to++){
		if(ants[pos].visited[to] == 0){
			denom += antProduct( from, to );
      //printf("%lf -- denom\n", denom);
		}
	}
   
  assert(denom != 0.0); 

  int count = CITIES;

	do{
		double p = 0.0;
		to++;

		if(to >= CITIES)
			to=0;

		if(ants[pos].visited[to] == 0){
			p = (double) antProduct(from,to)/denom;

			double x =  ((double)rand()/(double)RAND_MAX); 
      //printf("Denon: %18.50f -- X: %18.50f, p: %18.50f\n",denom, x,p);
			if(x < p){
        //printf("%lf -- X\n", x);
				break;
			}
      count--;
      if(count == 0){
        break;
      }
		}//sleep(3);
	}while(1);

	return to;
}

int simulate_ants(){
  int k, moving = 0; 
  
  for(k = 0; k < ANTS; k++){ 
    //printf("Formiga (%i)\n", k);
	if( ants[k].pathIndex < CITIES ){ //check if all cities were visited
		ants[k].nextCity = NextCity(k);
		ants[k].visited[ants[k].nextCity] = 1;
		ants[k].path[ants[k].pathIndex++] = ants[k].nextCity;

		ants[k].tourLength += distances[ants[k].curCity][ants[k].nextCity];

		//handle last case->last city to first

		if(ants[k].pathIndex == CITIES){
			ants[k].tourLength += distances[ants[k].path[CITIES -1]][ants[k].path[0]];
		}

		ants[k].curCity = ants[k].nextCity;
		moving++;

	}
  } 
  return moving;
}


void move_ants(){
  int curtime = 0;

  while(curtime++ < MAX_TIME){ 
    if( simulate_ants() == 0){
      //  Updating the trails of the ants
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
	


      if (curtime != MAX_TIME)
          restart_ants();
    }
    if(MAX_TIME%curtime == 0 )
    {
      printf("Best: %lf\n", bestdistance);
  	}
  }
}


int main()
{
 printf("%i -", MAX_TOTAL_DISTANCE);
  get_distances_matrix();

  initialize_ants();
  
  srand(time(NULL));
  printf("End - setting data; Begin -- calculations\n");

  move_ants();
  
  
 
	return 0;
}
