// Xerus - A General Purpose Tensor Library
// Copyright (C) 2014-2018 Benjamin Huber and Sebastian Wolf. 
// 
// Xerus is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License,
// or (at your option) any later version.
// 
// Xerus is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
// 
// You should have received a copy of the GNU Affero General Public License
// along with Xerus. If not, see <http://www.gnu.org/licenses/>.
//
// For further information on Xerus visit https://libXerus.org 
// or contact us at contact@libXerus.org.

#include <fstream>
#include <string>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <set>

#include <boost/filesystem.hpp>

#include "include/xerus.h"

std::mt19937_64 rnd = xerus::misc::randomEngine;
std::normal_distribution<double> normalDist(0,1);

using namespace xerus;

// ---------------------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------- general settings --------------------------------------------------------------

const value_t HISTOGRAM_BASE_CONVERGENCE_RATES = 1.2;
const value_t HISTOGRAM_BASE_END_RESIDUAL = 1.7;
const size_t NUM_SOLVES_PER_RUN = 10;

// ---------------------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------- benchmark problems ------------------------------------------------------------

using LeastSquaresSolver = std::pair<std::string, std::function<double(const TTOperator&, TTTensor&, const TTTensor&, PerformanceData&)>>;

struct LeastSquaresProblem {
	std::string name;
	std::vector<size_t> dimensions;
	std::vector<size_t> x_ranks;
	std::vector<size_t> b_ranks;
	std::vector<LeastSquaresSolver> solver;
	LeastSquaresProblem(const std::string &_name, const std::vector<LeastSquaresSolver> &_solver)
		: name(_name), solver(_solver) {};
	
	virtual TTOperator get_a() const {
		std::vector<size_t> dim(dimensions);
		dim.insert(dim.end(), dimensions.begin(), dimensions.end());
		return TTOperator::identity(dim);
	}
	virtual TTTensor get_x() const {
		TTTensor x = TTTensor::random(dimensions, x_ranks, normalDist);
		x /= frob_norm(x);
		return x;
	};
	virtual TTTensor get_b() const {
		TTTensor b = TTTensor::random(dimensions, b_ranks, normalDist);
		b /= frob_norm(b);
		return b;
	};
};

namespace ls {
	struct approximation : public LeastSquaresProblem {
		approximation(size_t _n, size_t _d, size_t _rankB, size_t _rankX, const std::vector<LeastSquaresSolver> &_solver)
			: LeastSquaresProblem("approximation", _solver)
		{
			dimensions = std::vector<size_t>(_d, _n);
			x_ranks = std::vector<size_t>(_d-1, _rankX);
			b_ranks = std::vector<size_t>(_d-1, _rankB);
		};
	};
	
	struct random : public LeastSquaresProblem {
		std::vector<size_t> a_ranks;
		
		random(size_t _n, size_t _d, size_t _rankA, size_t _rankB, size_t _rankX, const std::vector<LeastSquaresSolver> &_solver)
			: LeastSquaresProblem("random", _solver)
		{
			dimensions = std::vector<size_t>(_d, _n);
			a_ranks = std::vector<size_t>(_d-1, _rankA);
			x_ranks = std::vector<size_t>(_d-1, _rankX);
			b_ranks = std::vector<size_t>(_d-1, _rankB);
		};
		
		TTOperator get_a() const override {
			std::vector<size_t> dim(dimensions);
			dim.insert(dim.end(), dimensions.begin(), dimensions.end());
			TTOperator A = TTOperator::random(dim, a_ranks, normalDist);
			A /= frob_norm(A);
			return A;
		}
	};
	
	struct symmetric_posdef_random : public LeastSquaresProblem {
		std::vector<size_t> a_ranks;
		
		symmetric_posdef_random(size_t _n, size_t _d, size_t _rankA, size_t _rankB, size_t _rankX, const std::vector<LeastSquaresSolver> &_solver)
			: LeastSquaresProblem("symmetric_posdef_random", _solver)
		{
			dimensions = std::vector<size_t>(_d, _n);
			a_ranks = std::vector<size_t>(_d-1, _rankA);
			x_ranks = std::vector<size_t>(_d-1, _rankX);
			b_ranks = std::vector<size_t>(_d-1, _rankB);
		};
		
		TTOperator get_a() const override {
			std::vector<size_t> dim(dimensions);
			dim.insert(dim.end(), dimensions.begin(), dimensions.end());
			TTOperator A = TTOperator::random(dim, a_ranks, normalDist);
			Index i,j,k;
			A(i,j) = A(i,k) * A(j,k);
			A /= frob_norm(A);
			return A;
		}
	};
}


std::vector<LeastSquaresSolver> leastSquaresAlgorithms{
	{"ALS", ALSVariant(1, 0, 1e-8, ALSVariant::lapack_solver, true)}, 
	{"CG", GeometricCGVariant(0, 0, 1e-8, false, SubmanifoldRetractionI, ProjectiveVectorTransport)}, 
	{"SteepestDescent_submanifold", SteepestDescentVariant(0, 1e-8, false, SubmanifoldRetractionII)},
	{"SteepestDescent_als", SteepestDescentVariant(0, 1e-8, false, ALSRetractionII)},
	{"SteepestDescent_hosvd", SteepestDescentVariant(0, 1e-8, false, HOSVDRetraction(3))}, //TODO
};

std::vector<LeastSquaresSolver> leastSquaresAlgorithmsSPD{
	{"ALS", ALSVariant(1, 0, 1e-8, ALSVariant::lapack_solver, true)}, 
	{"CG", GeometricCGVariant(0, 0, 1e-8, true, SubmanifoldRetractionI, ProjectiveVectorTransport)}, 
	{"SteepestDescent_submanifold", SteepestDescentVariant(0, 1e-8, true, SubmanifoldRetractionII)},
	{"SteepestDescent_als", SteepestDescentVariant(0, 1e-8, true, ALSRetractionII)},
	{"SteepestDescent_hosvd", SteepestDescentVariant(0, 1e-8, true, HOSVDRetraction(3))}, //TODO
};

struct Approximation_Variant {
	std::function<double(TTTensor&, const TTTensor&, PerformanceData&)> solver;
	Approximation_Variant(std::function<double(TTTensor&, const TTTensor&, PerformanceData&)> _solver)
		: solver(_solver) {}
	double operator()(const TTOperator&, TTTensor &_x, const TTTensor &_b, PerformanceData &_pd) const {
		return solver(_x, _b, _pd);
	}
};

std::vector<LeastSquaresProblem> leastSquaresProblems{
	ls::approximation(2, 10, 4, 2, std::vector<LeastSquaresSolver>{
		{"ALS", Approximation_Variant(ALSVariant(1, 0, 1e-8, ALSVariant::lapack_solver, true))}, 
		{"CG", Approximation_Variant(GeometricCGVariant(0, 0, 1e-8, true, SubmanifoldRetractionI, ProjectiveVectorTransport))}, 
		{"SteepestDescent_submanifold", Approximation_Variant(SteepestDescentVariant(0, 1e-8, true, SubmanifoldRetractionII))},
		{"SteepestDescent_als", Approximation_Variant(SteepestDescentVariant(0, 1e-8, true, ALSRetractionII))},
		{"SteepestDescent_hosvd", Approximation_Variant(SteepestDescentVariant(0, 1e-8, true, HOSVDRetraction(2)))}, //TODO
	}),
	ls::random(2, 10, 3, 3, 3, leastSquaresAlgorithms),
	ls::symmetric_posdef_random(2, 10, 2, 3, 3, leastSquaresAlgorithmsSPD)
};




// ---------------------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------- benchmark routines ------------------------------------------------------------


std::string generate_profile_name() {
	std::string profileName;
#ifdef XERUS_TEST_COVERAGE
	static_assert(false, "test coverage checking nonsensical with benchmark run");
#endif
	
#ifdef XERUS_VERSION
	profileName += STRINGIFY(XERUS_VERSION);
#else
	profileName += "unknownVersion";
#endif
	
#ifdef LOW_OPTIMIZATION
	profileName += "_lowOpt";
#elif defined(HIGH_OPTIMIZATION)
	profileName += "_highOpt";
#elif defined(DANGEROUS_OPTIMIZATION)
	profileName += "_dangerousOpt";
#elif defined(RIDICULOUS_OPTIMIZATION)
	profileName += "_ridiculousOpt";
#else
	profileName += "_noOpt";
#endif
	
#ifdef USE_LTO
	profileName += "_lto";
#endif
#ifdef XERUS_DISABLE_RUNTIME_CHECKS
	profileName += "_noChecks";
#endif
#ifdef XERUS_REPLACE_ALLOCATOR
	profileName += "_replaceAlloc";
#endif
#ifdef XERUS_PERFORMANCE_ANALYSIS
	profileName += "_perfAnalysis";
#endif
	return profileName;
}


int main() {
	std::string profileName(generate_profile_name());
	LOG(benchmark, "running profile " << profileName);
	while (true) {
		for (const LeastSquaresProblem &prob : leastSquaresProblems) {
			std::vector<TTOperator> A;
			std::vector<TTTensor> X;
			std::vector<TTTensor> B;
			for (size_t i=0; i< NUM_SOLVES_PER_RUN; ++i) {
				A.emplace_back(prob.get_a());
				X.emplace_back(prob.get_x());
				B.emplace_back(prob.get_b());
			}
			for (const LeastSquaresSolver &solver : prob.solver) {
				LOG(benchmark, "solving " << prob.name << " with " << solver.first);
				PerformanceData perfData;
				misc::LogHistogram speedHist(HISTOGRAM_BASE_CONVERGENCE_RATES);
				misc::LogHistogram residualHist(HISTOGRAM_BASE_END_RESIDUAL);
				
				for (size_t i=0; i< NUM_SOLVES_PER_RUN; ++i) {
					perfData.reset();
					// solving the system
					TTTensor xCpy(X[i]);
					solver.second(A[i], xCpy, B[i], perfData);
					
					// generate histograms of this run
					speedHist += perfData.get_histogram(HISTOGRAM_BASE_CONVERGENCE_RATES, true);
					residualHist.add(perfData.data.back().residual);
				}
				
				// merge histograms with data on disk
				std::string fileName = std::string("benchmark/")+profileName+"/"+prob.name+"/"+solver.first;
				if (boost::filesystem::exists(fileName+"_speed.tsv")) {
					misc::LogHistogram speedHistFile = misc::LogHistogram::read_from_file(fileName+"_speed.tsv");
					speedHist += speedHistFile;
				} else {
					// will fail silently if the directories already exist
					boost::filesystem::create_directories(std::string("benchmark/")+profileName+"/"+prob.name);
				}
				speedHist.dump_to_file(fileName+"_speed.tsv");
				
				if (boost::filesystem::exists(fileName+"_residual.tsv")) {
					misc::LogHistogram residualHistFile = misc::LogHistogram::read_from_file(fileName+"_residual.tsv");
					residualHist += residualHistFile;
				} else {
					// will fail silently if the directories already exist
					boost::filesystem::create_directories(std::string("benchmark/")+profileName+"/"+prob.name);
				}
				residualHist.dump_to_file(fileName+"_residual.tsv");
			}
		}
	}
}
