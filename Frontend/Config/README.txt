This folder contains configuration files for the Analogsat Frontend
to (re-)create the results shown in the paper
'Accelerating a continuous-time SAT solver using GPUs'
by F. Molnar, S. R. Kharel, X. Hu, Z. Toroczkai.

The configuration files are simply a list of command-line options that
can also be entered manually when launching analogsat. In the configuration
files each line holds one option, and the first space character in the line
separates the option from its argument value (hence, for options that
accept string values, the strings may contain spaces).

If the configuration file option is followed by additional options in the
command line, then those settings will override the ones in the configuration
files. In essence, the last value of any option (if specified multiple times)
will be used. 

For example,
analogsat bench -config Config/make_340_sat_1.cfg -timeout 7200
will use timeout=7200, despite the setting in the .cfg file, which sets the
timeout to 3600.

The purpose of each configuration is listed below.
Also, for each configuration, the approximate run time is listed.

1. Trajectory data for illustration 
===================================

traj_orig.cfg - trajectory data using the original CTDS
traj_tanh.cfg - trajectory data using the alternative (Tanh-based) CTDS
(runtime: 1 minute)

2. Problem generation
=====================

These configurations create random SAT problem samples that are used later
for runtime and speedup measurements. The objective here is to verify as 
quickly as possible the satisfiability of each problem. Problems that are 
found to be satisfiable will be saved as cnf. Those that remain unsolved 
(unsatisfiable or just not solved within timeout) will not be saved. 
Existing problems will not be overwritten, unless directed to do so.

IMPORTANT: Since unsolved problems are not saved, the repeated running of
problem generation can fill these missing problems. This is by design, 
but it is also a source of a huge CAVEAT if used incorrectly! 

In particular, if unsaved problems are replaced because those problems were 
indeterminate (ran out of walltime limit), then it would introduce a BIAS
toward problems that *can* be solved in the given walltime limit (i.e.,
they are in fact easier problems than the desired problem class). The
repeated use of a configuration is only valid when it is being used to 
replace UNSAT problems, not indeterminate ones!

There are two implications. First, if the walltime limit is reached on a
class of problems, those problems must be completely discarded, the walltime
limit increased, and the sample generatinon repeated. This applies to both
AnalogSat and MiniSat.

The second implication is that AnalogSat cannot be used for problem 
verification when the clause ratio alpha is near the SAT-UNSAT transition
point, because AnalogSat is an incomplete solver. For UNSAT problems,
AnalogSat always runs endlessly, up to any walltime limit. However, given
enough time, MiniSat (being a complete solver) can report either SAT or
UNSAT. Only at this time, when the problems are SAT or UNSAT, but not
indeterminate, is the time when repeated running of the problem generation
is allowed, as the objective is to find satisfiable, but otherwise unbiased
samples, for the benchmarks.


Below, the approximate run times are also indicated for each configuration.

2.1. Generate SAT samples for alpha = 3.4
-----------------------------------------
make_340_sat_1.cfg : 10 minutes
make_340_sat_2.cfg : 10 minutes

2.2. Generate SAT samples for alpha = 3.8
-----------------------------------------
make_380_sat_1.cfg : 2 minutes
make_380_sat_2.cfg : 10 hours
make_380_sat_3.cfg : 7 hours

2.3. Generate SAT samples for alpha = 4.25 
------------------------------------------
make_425_sat_1.cfg : 1 hour
make_425_sat_p1.cfg .. make_425_sat_p8.cfg  
    (parallel runs, heavy computation): about 5 hours each

2.4. Problems too large to solve but used for speedup measurement
-----------------------------------------------------------------

Here the solver runs only for one iteration (checks that the problem fits 
in memory), the cnf file is forced to be written, and there is only one 
sample per problem size.

make_340_large.cfg : 1 minute
make_380_large.cfg : 1 minute
make_425_large.cfg : 1 minute

2.5. Special-size problems for speedup measurement only
-------------------------------------------------------
		
make_2000_special.cfg : 1 minute
make_5000_special.cfg : 2 minutes


3. Solving time benchmarks
==========================

Here the walltime limits should be sufficiently large to allow for any
(reasonable) solving time. Limits are based on the run time allowance
on the clusters where we run them.

3.1. alpha = 3.4 
----------------

nEnd: 9.0 for minisat, 12.0 for analogsat

bench_340_run_mini.cfg: 78 hrs
bench_340_run_analog.cfg: 8 hrs
bench_340_run_tanh.cfg: 5 hrs

3.2. alpha = 3.8 
----------------

nEnd: 6.5 for minisat, 8.5 for analogsat

bench_380_run_mini.cfg: 22 hrs
bench_380_run_analog.cfg 45 hrs
bench_380_run_tanh.cfg: 12 hrs

3.3. alpha = 4.25 
-----------------

nEnd: 5.5 for both solvers

bench_425_run_mini.cfg: 41 hrs
bench_425_run_analog.cfg 46 hrs
bench_425_run_tanh.cfg: 31 hrs


4. AnalogSat CPU/GPU Speedup
==============================

Each speedup configuration takes about 5 minutes to compute.

4.1. alpha = 3.4
----------------

speedup_340_cpu.cfg
speedup_340_gpu1.cfg
speedup_340_gpu2.cfg
speedup_340_gpu3.cfg

4.2. alpha = 3.8
----------------

speedup_380_cpu.cfg
speedup_380_gpu1.cfg
speedup_380_gpu2.cfg
speedup_380_gpu3.cfg

4.3. alpha = 4.25
----------------

speedup_425_cpu.cfg
speedup_425_gpu1.cfg
speedup_425_gpu2.cfg
speedup_425_gpu3.cfg

4.4. alpha = 20.0 with K=6
--------------------------

speedup_2000_cpu.cfg
speedup_2000_gpu1.cfg
speedup_2000_gpu2.cfg
speedup_2000_gpu3.cfg

4.5. alpha = 50.0 with K=10
---------------------------

speedup_5000_cpu.cfg
speedup_5000_gpu1.cfg
speedup_5000_gpu2.cfg
speedup_5000_gpu3.cfg



5. Ramsey problem generation
============================

Ramsey graph coloring problems are a great source of benchmarks.
Moreover, if one can find a solution to a Ramsey problem for a graph
larger than the known lower bound of the corresponding Ramsey number,
then the existence of the solution itself proves that the particular 
Ramsey number bound is higher than the previously known one.

For example, it is known that R(5,5) >= 43. The corresponding SAT problem
is ramsey_regular_55_N43.cnf. This is a very hard problem, and no solution
has ever been found (as of 2019). But if someone finds a solution, it would 
prove that R(5,5) >= 44. 

On the other hand, this also means that all ramsey_regular_55_N**.cnf problems
with N<=42 are satisfiable. Nonetheless, these are hard problems and it
would take a very long time to find solutions, without using additional 
tricks (specially designed initial conditions instead of random ones).


5.1. Problems for R(5,5)
------------------------

make_ramsey_55.cfg: 1 minute

5.2. Problems for R(3,3,3,3)
----------------------------

make_ramsey_3333.cfg: 1 minute



6. Ramsey problem benchmarks
============================

Ramsey problems are attempted using minisat, analogsat on GPU, and analogsat
using alternative (Tanh) CTDS on GPU. Problems start from N=20 nodes in the
graph, going up to the known limit. Each problem is repeated 10 times.
The runs must be eventually aborted, since it would take prohibitively long 
time to solve the largest problems by either solvers.

6.1. Problems for R(5,5)
------------------------

run_ramsey_55_mini.cfg: 60 hrs
run_ramsey_55_analog.cfg: 4 hrs
run_ramsey_55_tanh.cfg: 3 hrs


6.2. Problems for R(3,3,3,3)
----------------------------

run_ramsey_3333_mini.cfg: 55 hrs
run_ramsey_3333_analog.cfg: 30 mins
run_ramsey_3333_tanh.cfg: 19 hrs

