#ifndef FRONTEND_CNF_RUNNER_H
#define FRONTEND_CNF_RUNNER_H

#include <random>
#include "config.h"

// run a given SAT problem 
// also capable of exporting CTDS trajectory (for plotting purposes only, because doing so will be slow and inefficient)
void RunCnf(Configuration conf);

#endif
