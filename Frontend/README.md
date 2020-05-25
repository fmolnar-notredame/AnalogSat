# AnalogSAT Frontend

Copyright(C) 2019 Ferenc Molnar

License: GNU GPL v3.

Simulations of continuous-time dynamical systems (CTDS) that solve Boolean
satisfiability (SAT) problems or minimize the number of violated clauses.

This program can be used as a SAT solver on a given CNF input. In addition, 
pre-defined command configurations are available to re-create the results 
shown in the paper:
'Accelerating a continuous-time SAT solver using GPUs'
by F. Molnar, S. R. Kharel, X. Hu, Z. Toroczkai.

See the details of each configuration in Config/README.txt.


## Usage: 
`analogsat <command> [-option <value>]`

### Commands:
- `run`           Run a given SAT problem from a CNF file
- `bench`         Measure the solving time of a series of random SAT problems
- `speedtest`     Measure the iteration performance on a series of problems
- `make_ramsey`   Make CNF files representing Ramsey graph coloring problems
- `run_ramsey`    Run Ramsey graph coloring problems made by make_ramsey

See below for the options of each command. These may include parameters, and their possible values are indicated as follows:
  - allowed range of values: `(a::b)`
  - allowed set of values: `(a/b/c)`
  - default value if option is not present: `[a]`

Value types:
  - `<N>`: integer
  - `<F>`: floating-point
  - `<S>`: string
  - no type given: `boolean`, the option is `TRUE` if specified, `FALSE` otherwise.

## Common options for all commands

### Solver parameters

- `-version <N>`  AnalogSAT GPU solver version `(1/2/3) [1]`
- `-nogpu`        Use the AnalogSAT CPU solver, do not call any GPU functions
- `-minisat`      Use the MiniSat solver (CPU), do not call any GPU functions
- `-usegpu <N>`   Use CUDA device number N (via CudaSetDevice(N)) `(0::99) [0]`
- `-tanh`         Use the alternative CTDS formulation, which is based on the
              Tanh formula for evolving the auxiliary variables.
- `-bias <F>`     Coefficient for the bias term in the CTDS `(0::1e6) [0]`

### ODE Integration parameters

- `-tmax <F>`     Maximum analog time for the integration `(0::1e80) [1e8]`
- `-stempax <N>`  Maximum number of discrete steps to take in the integration.
              `(0::INT_MAX) [50000000]`
- `-timeout <F>`  Walltime limit for the integration, in seconds `(0::1e80) [3600]`
- `-eps <F>`      Relative error tolerance parameter for the adaptive time stepping
              of the ODE integrator method `(1e-8::1e-1) [1e-6]`
- `-batch <N>`    Number of discrete steps to take at once before the current state
              is checked for a SAT solution `(1::INT_MAX) [50]`

## The `run` command


2.1. Description
----------------

Runs a given SAT problem from a CNF file.
Input file must be in DIMACS CNF format.
Output file will be compatible with minisat's output format.

File naming: the CNF file name (without extension and path) is the problem's
name, which is used as prefix for all output files for a given problem.

For example, given the input
/some/where/abcde.cnf,
the outputs will be:
<resultfolder>/abcde.out  for the SAT solution,
<resultfolder>/abcde_traj_<solverfamily>.dat  for the trajectory (when enabled)

2.2. Options
------------
-problem <S>        Name of the input CNF file.
-resultfolder <S>   Folder where the results are saved. Folder is created if
                    it does not exist. [.]
-trajectory         Save the trajectory of the integrated CTDS. The trajectory
                    is saved at the resolution specified by -batch. 


3. The 'bench' command
----------------------

3.1. Description
----------------

Runs a series of random SAT problems and measures the solving time.

Problems are identified by their class (defined by number of variables and
clause ratio), and random samples within each class are identified by the
sample index.

If a problem with a given sample index is found satisfiable, the corresponding
CNF file is written to the <problemfolder>. This file will be loaded the next
time this sample index is requested. Saving (and overwriting existing) CNFs
can be forced regardless of satisfiability by -force_save_cnf.

Results are written to <resultfolder>, using the 'perf' prefix, and problem
class identifiers in the file name. See the Manual.pdf for details on content.

The number of variables in a problem is computed using the following formula:
N = 10 * (2^n),
where n is a parameter that varies in a loop, given via Options (see below).


3.2. Options
------------

-problemfolder <S>  Folder where the CNF files for the random problems are
                    saved. Subfolders will be created for problem classes 
                    automatically.
-resultfolder <S>   Folder where the results are saved.
-force_save_cnf     Save the CNF of the current sample to the problemfolder
                    regardless of satisfiability.
-samplestart <N>    Start index for problem samples, inclusive (0::9999) [0]
-sampleend <N>      End index for problem samples, exclusive (0::10000) [100]
-k <N>              The length of clauses made in random problems (2::256) [3]
-alpha <F>          The ratio of clauses made, relative to the number of
                    variables in the problem (0::1e10) [4.25]
-nstart <F>         Starting value of the problem size exponent, inclusive,
                    (1::100) [1.0]
-nend <F>           Ending value of the problem size exponent, inclusive,
                    (1::100) [5.0]
-nstep <F>          Step value of the problem size exponent (0::100) [0.5]
-rerun              Run the given sample, even if it has been solved before

4. The 'speedtest' command
--------------------------

4.1. Description
----------------

Measures the iteration performance on a series of problems. Iterations are
repeated up to 5 seconds walltime regardless of finding a solution or not. If
a solution is found, the problem is solved again from random initial 
conditions. Note, the -timeout option is respected for each individual run.

Problems must already exists for speedtest. To make them, use the 'bench'
command mode.


4.2. Options
------------
-problemfolder <S>  Folder from where the CNF files are loaded.
-resultfolder <S>   Folder where the results are saved.
-samplestart <N>    Start index for problem samples, inclusive (0::9999) [0]
-sampleend <N>      End index for problem samples, exclusive (0::10000) [100]
-alpha <F>          The ratio of clauses made, relative to the number of
                    variables in the problem (0::1e10) [4.25]
-nstart <F>         Starting value of the problem size exponent, inclusive,
                    (1::100) [1.0]
-nend <F>           Ending value of the problem size exponent, inclusive,
                    (1::100) [5.0]
-nstep <F>          Step value of the problem size exponent (0::100) [0.5]


5. The 'make_ramsey' command
----------------------------

5.1. Description
----------------

Creates a CNF representation for a series of Ramsey graph coloring problems.

A Ramsey problem asks if there is an edge coloring on a complete graph 
(clique) of N nodes such that there are no monochromatic cliques of given
sizes, for given colors, in the graph. The sizes of monochromatic cliques to 
be avoided for each color are given by integers. 

For example,
- R(5,5) asks for edge colorings of complete graphs using 2 colors, where
  no monochromatic cliques of 5 nodes exist using either color.
- R(4,6) asks for edge colorings using 2 colors, where no monochromatic clique
  of size 4 exist using the first color, nor monochromatic cliques of size 6 
  exist using the second color.
- R(4,5,6) asks for colorings using 3 colors, where no monochromatic clique of
  size 4 exists with the first color, no monochromatic cliques of size 5 exist
  with the second color, and no monochromatic cliques exist using the third
  color.
- R(3,3,3,3) asks for coloring using 4 colors, where no monochromatic cliques
  of size 3 (i.e., triangles) exist using either colors.

In the CNF representation, Boolean variables are assigned to the bits of the 
color values for each edge (1 bit for 2 colors, 2 bits for 3 or 4 colors, and
3 bits for up to 8 colors). The clauses express the forbidden monochromatic
cliques. Thus, solving such a SAT problem gives also a valid edge coloring.


5.2. Options
------------

-R <S>              Comma-separated list of integers that specify the
                    Ramsey problem. E.g., -R 3,4,5 for problem R(3,4,5).
-problemfolder <S>  Folder where the CNF files for Ramsey problems are saved.
-nstart <N>         Starting size N for the number of nodes in the graph.
-nend <N>           Ending size N for the number of nodes in the graph 
                    (inclusive).
-ramseycircular     Add the constraint that the adjacency matrix of the graph
                    is a circulant matrix. 

6. The 'run_ramsey' command
---------------------------

Runs the Ramsey problems created by the 'make_ramsey' command.
Saves the solving time, as well as the found graph coloring.


6.2. Options
------------

-R <S>              Comma-separated list of integers that specify the
                    Ramsey problem. E.g., -R 3,4,5 for problem R(3,4,5).
-problemfolder <S>  Folder where the CNF files for Ramsey problems are saved.
-resultfolder <S>   Folder where results are saved.
-nstart <N>         Starting size N for the number of nodes in the graph.
-nend <N>           Ending size N for the number of nodes in the graph 
                    (inclusive).
-ramseycircular     Add the constraint that the adjacency matrix of the graph
                    is a circulant matrix. 
