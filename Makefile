
FLAG = -std=c++11 -pthread

all:
	g++ PDP1_Likhovod_Kirill.cpp -o Kirill $(FLAG)

1:
	./a.out 1

0:
	./a.out 0 

clean:
	rm a.out

run:
	g++ PDP1_Likhovod_Kirill.cpp  $(FLAG)
	./a.out 1