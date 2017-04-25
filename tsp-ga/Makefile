CC = g++
NVCC = nvcc
CUDA_PATH = /usr/local/cuda
CFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lcuda -lcurand -lm  
NVCCFLAGS= -D_FORCE_INLINES -Xcompiler -fPIC  -I$(CUDA_SDK_PATH)/C/common/inc
COPTFLAGS = -O3 -g -std=c++11 
LDFLAGS =

all: 	main.o main_cpu.o world.o common.o ga_cpu.o ga_gpu.o tsp-ga-gpu tsp-ga-cpu

tsp-ga-gpu: main.o 
	$(CC) main.o world.o ga_gpu.o common.o  $(CFLAGS) $(COPTFLAGS) -o tsp-ga-gpu
tsp-ga-cpu: main_cpu.o 
	$(CC) main_cpu.o world.o ga_cpu.o common.o  $(CFLAGS) $(COPTFLAGS) -o tsp-ga-cpu

ga_gpu.o: ga_gpu.cu  world.h
	$(NVCC) -c ga_gpu.cu $(CFLAGS) $(COPTFLAGS) $(NVCCFLAGS)


ga_cpu.o: ga_cpu.cpp  world.h common.h
	$(NVCC) -c ga_cpu.cpp $(CFLAGS) $(COPTFLAGS)

common.o: common.cpp common.h  world.h
	$(NVCC) -c common.cpp $(CFLAGS) $(COPTFLAGS)

world.o: world.cpp world.h
	$(NVCC) -c world.cpp $(CFLAGS) $(COPTFLAGS) $(NVCCFLAGS)
main.o: main.cpp  common.h ga_gpu.h  world.h
	$(NVCC) -c main.cpp $(CFLAGS) $(COPTFLAGS) $(NVCCFLAGS)
main_cpu.o: main_cpu.cpp  common.h ga_cpu.h  world.h
	$(NVCC) -c main_cpu.cpp $(CFLAGS) $(COPTFLAGS) 
