
# CUDA includes and libraries



# F1= -L/usr/local/cuda/lib64
F1= -L/usr/local/cuda-10.1/lib64


# F2= -I/usr/local/cuda-9.2/targets/x86_64-linux/include -lcuda -lcudart
F2= -I/usr/local/cuda-10.1/targets/x86_64-linux/include -lcuda -lcudart

# SDL shtuff (for sound processing)
#F3= -I/usr/local/include -L/usr/local/lib -lSDL2
F4= -std=c++11 -pthread
#F4= -std=c++14

# animation libraries:
F5= -lglut -lGL


all: PDP2_LikhovodKirill

PDP2_LikhovodKirill: interface.o animate.o gpu_main.o attractor.o visualizeAttractor.o
	g++ -o PDP2_LikhovodKirill interface.o gpu_main.o animate.o attractor.o visualizeAttractor.o $(F1) $(F2) $(F3) $(F4) $(F5)

# do we really need all these flags to compile interface??!!
interface.o: interface.cpp gpu_main.cu gpu_main.h animate.h animate.cu attractor.h visualizeAttractor.h
	g++ -w -c interface.cpp $(F1) $(F2) $(F3) $(F4)

attractor.o: attractor.cpp attractor.h
	g++ -c attractor.cpp $(F1) $(F2) $(F3) $(F4)

visualizeAttractor.o: visualizeAttractor.cpp gpu_main.cu animate.cu
	g++ -w -c visualizeAttractor.cpp $(F1) $(F2) 

gpu_main.o: gpu_main.cu gpu_main.h
	nvcc -w -c gpu_main.cu $(F1) $(F2) $(F3) 

animate.o: animate.cu animate.h gpu_main.h
	nvcc -w -c animate.cu $(F1) $(F2) $(F3) 
#	nvcc -w -c animate.cu $(ANIMLIBS)

#audio.o: audio.c audio.h
#	g++ -w -c audio.c $(F2)

run:  all 
	./PDP2_LikhovodKirill -r2



clean:
	rm interface.o;
	rm animate.o;
	rm gpu_main.o;
	rm PDP2_LikhovodKirill;
	rm visualizeAttractor.o;
	rm attractor.o;
