#!/bin/bash

echo Compiling and Linking frontend
nvcc -std=c++11 -rdc=true -O3 -L../AnalogSAT -I../AnalogSAT/include -I../MiniSat/include -L../MiniSat -lanalogsat -lminisat -lcurand -lcudart_static -o analogsat src/*.cpp
