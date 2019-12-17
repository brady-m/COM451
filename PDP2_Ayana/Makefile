
# CUDA includes and libraries
# F1= -L/usr/local/cuda/lib64
F1= -L/usr/local/cuda-10.1/lib64
F2= -I/usr/local/cuda-9.2/targets/x86_64-linux/include -lcuda -lcudart
# F2= -I/usr/local/cuda-10.1/targets/x86_64-macos/include -lcuda -lcudart

# SDL shtuff (for sound processing)
#F3= -I/usr/local/include -L/usr/local/lib -lSDL2
F4= -std=c++11
#F4= -std=c++14

# animation libraries:
F5= -lglut -lGL

all: MyViz

MyViz: interface.o animate.o gpu_main.o
	g++ -o MyViz interface.o gpu_main.o animate.o $(F1) $(F2) $(F3) $(F4) $(F5)

# do we really need all these flags to compile interface??!!
interface.o: interface.cpp gpu_main.cu gpu_main.h animate.h animate.cu
	g++ -w -c interface.cpp $(F1) $(F2) $(F3) $(F4)

gpu_main.o: gpu_main.cu gpu_main.h
	nvcc -w -c gpu_main.cu

animate.o: animate.cu animate.h gpu_main.h
	nvcc -w -c animate.cu
#	nvcc -w -c animate.cu $(ANIMLIBS)

#audio.o: audio.c audio.h
#	g++ -w -c audio.c $(F2)

clean:
	rm interface.o;
	rm animate.o;
	rm gpu_main.o;
	rm MyViz;
