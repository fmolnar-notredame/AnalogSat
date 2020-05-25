#!/bin/bash
GEN1=arch=compute_60,code=sm_60
GEN2=arch=compute_61,code=sm_61
GEN3=arch=compute_70,code=sm_70
echo $GEN1
echo $GEN2
echo $GEN3

rm -f *.o

echo Compiling cuda_base
nvcc -rdc=true -gencode $GEN1 -gencode $GEN2 -gencode $GEN3 -std=c++11 -O3 -c cuda_base/*.cu

echo Compiling cuda_sat
nvcc -rdc=true -gencode $GEN1 -gencode $GEN2 -gencode $GEN3 -std=c++11 -O3 -c cuda_sat/*.cu

echo Compiling cpu
nvcc -rdc=true -gencode $GEN1 -gencode $GEN2 -gencode $GEN3 -std=c++11 -O3 -c cpu_sat/*.cpp

echo Compiling util
nvcc -rdc=true -gencode $GEN1 -gencode $GEN2 -gencode $GEN3 -std=c++11 -O3 -c util/*.cpp

echo Compiling io
nvcc -rdc=true -gencode $GEN1 -gencode $GEN2 -gencode $GEN3 -std=c++11 -O3 -c io/*.cpp

echo Compiling problem
nvcc -rdc=true -gencode $GEN1 -gencode $GEN2 -gencode $GEN3 -std=c++11 -O3 -c problem/*.cpp

echo Compiling solver
nvcc -rdc=true -gencode $GEN1 -gencode $GEN2 -gencode $GEN3 -std=c++11 -O3 -c solver/*.cpp

echo Static linking with CUDA
nvcc -dlink -gencode $GEN1 -gencode $GEN2 -gencode $GEN3 -o analogsat.oo *.o -lcudart_static -lcurand

echo Building library
ar cru libanalogsat.a *.o analogsat.oo
ranlib libanalogsat.a

rm *.o
rm *.oo
