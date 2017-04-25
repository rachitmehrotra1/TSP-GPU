CC = gcc
NVCC = nvcc
CUDA_PATH = /usr/local/cuda
CFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lcuda -lcurand -lm  
NVCCFLAGS= -D_FORCE_INLINES -Xcompiler -fPIC  -I$(CUDA_SDK_PATH)/C/common/inc
COPTFLAGS = -O3 -g -std=c++11 
LDFLAGS =

all:  ants.o ant_gpu.o tsp-ant-gpu tsp-ant-cpu

tsp-ant-gpu: ant_gpu.o
	$(NVCC)   $(CFLAGS) $(COPTFLAGS) parallel_ants.cu  -o tsp-ant-gpu
tsp-ant-cpu: ants.o
	$(CC) $(CFLAGS) ants.c -o tsp-ant-cpu

ant_gpu.o: parallel_ants.cu
	$(NVCC) -c parallel_ants.cu $(CFLAGS) $(COPTFLAGS) $(NVCCFLAGS)
ant_cpu.o: ants.c
	$(CC) -c ants.c $(CFLAGS) $(COPTFLAGS) $(NVCCFLAGS)