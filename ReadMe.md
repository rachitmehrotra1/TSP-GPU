ReadMe

Travel Salesman Problem using Genetic Algorithm & Ant Colony Optimization
Rachit Mehrotra
Project for CSCI-GA 3033 - 004

The code is divided into 2 part - TSP using GA and TSP using ACO

Instructions for TSP using GA

- The makefile consists of all the compilation instructions, thus just running 'make' would build everything necessary.All the code was built and tested on cuda2 and the run times in the report are also observed on cuda2 (sometimes device 0, sometimes device 2)
Note: If someone else is using the GPU , the run times could vary by 10-20%
- The application generates its own city data. Thus to run the code with different cities,population,max generation combination. You can make said changes in the main_cpu.cpp - line 28 (for the sequential code)/ main.cpp - line 29 (for the CUDA code). The files are well commented and it's really easy to make changes to the parameter. Just change those parameters and do 'make'.
- Then the sequential version can be run using ./tsp-ga-cpu and Cuda version using ./tsp-ga-gpu

Note, the main file also contains the 'seed' for the random number generator. Thus to make sure the CPU and GPU versions of the code are running on the same data, make sure that the seeds are same in both the main file. I have set them to be same , but in case you want to play around by trying different data.


Instructions for TSP using ACO
-For ACO , since it's a much smaller and simpler code , I just used 1 file each for parallel and CUDA version. I'm using an open source map_generator (coded in ruby), that takes the number of cities as a parameter and builds a map.txt that contains a random city map with said N cities. 
Command to run the map generator : ruby map_generator.rb Num_of_cities
-I have compiled and saved 3 different variants of maps for the ease of the grader to check my code. map25.txt , map50.txt , and map100.txt contains maps with 25,50,100 cities respectively.

-To run the sequential and parallel version of the code. Just do a 'make'
and to run the sequential version using for example to run for 25 cities -> ./tsp-ant-cpu < map25.txt
and to run the parallel version using for example 25 cities -> ./tsp-ant-gpu < map25.txt
This ensures that the input data for parallel and sequential version is same

NOTE: Just like GA, even in this, to play around with a number of cities, just open ants.c - line 7 & parallel_ants.cu - line 8and change the #define cities 25 , to whatever value you want ,25,50 or 100 and do a 'make' to compile the code. Since all the memory allocation and some other global variables depend on the 'CITIES' variable, I didn't take it as a parameter and instead defined it as a #define.


Happy Testing :) 