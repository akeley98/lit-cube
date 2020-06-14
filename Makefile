lit-cube: main.cc glad/src/glad.c gl_core_3_3.h
	clang++ -std=c++14 -O2 -Wall -Wextra glad/src/glad.c main.cc -g -I /usr/include/GL -I ./glad/include -ldl -lGL -lGLEW -lSDL2 -o lit-cube

