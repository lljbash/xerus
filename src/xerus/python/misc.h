#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wundef"
#pragma GCC diagnostic ignored "-Wfloat-equal"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wuseless-cast"
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/eval.h>
// #include <pybind11/numpy.h>
// #undef NDEBUG
#include "xerus.h"
#include "xerus/misc/internal.h"  // REQUIRE
#pragma GCC diagnostic pop


using namespace pybind11;
using namespace xerus;


#define parametersDocstr "\n\nParameters\n----------\n"
#define returnsDocstr "\n\nReturns\n-------\n"


void expose_indexedTensors(module& m);
void expose_factorizations(module& m);

void expose_tensor(module& m);
void expose_tensorNetwork(module& m);
void expose_ttnetwork(module& m);
void expose_htnetwork(module& m);

void expose_leastSquaresAlgorithms(module& m);
void expose_recoveryAlgorithms(module& m);

void expose_misc(module& m);
