#pragma once
// Floating-point type used everywhere. Timing builds use float; correctness
// tests are built with -DREAL_T=double so that summation-order differences
// stay near 1e-15 and anything larger points to a bug (plan D4).
#ifndef REAL_T
#define REAL_T float
#endif

using real = REAL_T;
