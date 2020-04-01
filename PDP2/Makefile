
# CUDA includes and libraries
# F1= -L/usr/local/cuda-9.2/lib64
# F2= -I/usr/local/cuda-9.2/targets/x86_64-linux/include -lcuda -lcudart
F1= -L/usr/local/cuda-10.1/lib64
F2= -I/usr/local/cuda-10.1/targets/x86_64-linux/include -lcuda -lcudart

# SDL shtuff (for sound processing)
#F3= -I/usr/local/include -L/usr/local/lib -lSDL2

F4= -std=c++11
#F4= -std=c++14

# animation libraries:
F5= -lglut -lGL

# thread flag
F6 = -pthread

all: PDP2_Umarbaev

PDP2_Umarbaev: PDP2_Umarbaev.cpp interface.o animate.o gpu_main.o cudaDeviceProp.h crack.h
	g++ -w -o PDP2_Umarbaev interface.o gpu_main.o animate.o PDP2_Umarbaev.cpp $(F1) $(F2) $(F3) $(F4) $(F5) $(F6)

# do we really need all these flags to compile interface??!!
interface.o: interface.h interface.cpp gpu_main.h gpu_main.cu animate.h animate.cu struct.h
	g++ -w -c interface.cpp $(F1) $(F2) $(F3) $(F4)

gpu_main.o: gpu_main.cu gpu_main.h struct.h # interface.h
	nvcc -w -c gpu_main.cu

animate.o: animate.cu animate.h gpu_main.h
	nvcc -w -c animate.cu
#	nvcc -w -c animate.cu $(ANIMLIBS)

#audio.o: audio.c audio.h
#	g++ -w -c audio.c $(F2)

clean:
	rm animate.o;
	rm gpu_main.o;
	rm interface.o;
	rm PDP2_Umarbaev;
