lit-cube: main.cc gl_core_3_3.c gl_core_3_3.h
	clang++ -std=c++14 -O2 -Wall -Wextra gl_core_3_3.c main.cc -g -I /usr/include/GL -lGL -lGLEW -lSDL2 -o lit-cube

