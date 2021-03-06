# AnalogSat
Simulations of continuous-time dynamical systems (CTDS) that solve Boolean satisfiability (SAT) problems or minimize the number of violated clauses (MAXSAT).

Corresponding publication:  
"_Accelerating a continuous-time SAT solver using GPUs_",  
F. Molnar, S. R. Kharel, X. S. Hu, Z. Toroczkai,  
Computer Physics Communications, Volume 256, November 2020, 107469.  
DOI: [https://doi.org/10.1016/j.cpc.2020.107469](https://doi.org/10.1016/j.cpc.2020.107469)

This work was supported in part by the National Science Foundation under Grants CCF-1644368, 1640081 and CNS1629914, and by the Nanoelectronics Research Corporation, a wholly-owned subsidiary of the Semiconductor
Research Corporation, through Extremely Energy Efficient Collective
Electronics, an SRC-NRI Nanoelectronics Research Initiative under Research
Task ID 2698.004.

This software package contains 3 separate modules:

- (1) AnalogSat library
- (2) MiniSat for AnalogSat library
- (3) AnalogSat Frontend

Each module can be compiled and used on its own. 
Each module is licensed under GNU GPL v3. 
See the license file in the folders of each module.

The code needs NVIDIA CUDA Toolkit (minimum version 9.2) to compile, but its CPU-based solvers can
be used without a CUDA-compatible GPU.

The AnalogSat library (1) is the core implementation of the Boolean 
satisfiability (SAT) solver described in the paper.

The MiniSat for AnalogSat library (2) is a modified version of the original 
MiniSat software by Niklas Een and Niklas Sorensson. It is based on the latest
available version of MiniSat (version 2.2.0) on http://minisat.se. The purpose
of the modifications is to make the MiniSat code accessible via the AnalogSat
Frontend. In addition, linux-specific parts (linux-specific timing and
interrupts) have been removed. The solving algorithm has not been modified.

The AnalogSat Frontend (3) is a command-line interface to run SAT problems through the solver. It also comes with predefined configurations that allow the user to recreate the results shown in the paper.
