echo Compiling Minisat
nvcc -rdc=true -std=c++11 -O3 -c core/*.cpp
nvcc -rdc=true -std=c++11 -O3 -c utils/*.cpp

echo Static linking 
nvcc -dlink -o minisat.oo *.o

echo Building library
ar cru libminisat.a *.o minisat.oo
ranlib libminisat.a

rm *.o
rm *.oo
