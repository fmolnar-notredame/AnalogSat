# AnalogSat
Simulations of continuous-time dynamical systems (CTDS) that solve Boolean satisfiability (SAT) problems or minimize the number of violated clauses.

This software package contains 3 separate modules:

- (1) AnalogSat library
- (2) MiniSat for AnalogSat library
- (3) AnalogSat Frontend

Each module can be compiled and used on its own. 
Each module is licensed under GNU GPL v3. 
See the license file in the folders of each module.

The AnalogSat library (1) is the core implementation of the Boolean 
satisfiability (SAT) solver described in the paper 

"_Accelerating a continuous-time SAT solver using GPUs_"
by F. Molnar, S. R. Kharel, X. Hu, Z. Toroczkai.
DOI: _to be added_

The code needs NVIDIA CUDA Toolkit to compile, but its CPU-based solvers can
be used without a CUDA-compatible GPU.

The MiniSat for AnalogSat library (2) is a modified version of the original 
MiniSat software by Niklas Een and Niklas Sorensson. It is based on the latest
available version of MiniSat (version 2.2.0) on http://minisat.se. The purpose
of the modifications is to make the MiniSat code accessible via the AnalogSat
Frontend. In addition, linux-specific parts (linux-specific timing and
interrupts) have been removed. The solving algorithm has not been modified.
