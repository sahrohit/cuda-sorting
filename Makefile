all: thrust singlethread multithread

thrust: thrust.cu
	nvcc thrust.cu -o thrust
	
singlethread: singlethread.cu
	nvcc singlethread.cu -o singlethread

multithread: multithread
	nvcc multithread.cu -o multithread
