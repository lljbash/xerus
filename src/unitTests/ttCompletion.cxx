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


#include<xerus.h>

#include "../../include/xerus/test/test.h"
#include "../../include/xerus/misc/internal.h"
#include "../../include/xerus/examples/tensorCompletion.h"
using namespace xerus;


static misc::UnitTest alg_adf_inverseidx("Algorithm", "adf_inverse_index_ratios", [](){
	const size_t D = 6;
	const size_t N = 10;
	const size_t R = 3;
	const size_t CS = 3;
	
	std::mt19937_64 &rnd = xerus::misc::randomEngine;
	std::uniform_int_distribution<size_t> dist(0, N-1);
	std::uniform_real_distribution<value_t> distF(-1.0, 1.0);
	SinglePointMeasurementSet measurements;
	
	REQUIRE(D*N*CS*R*R < misc::pow(N, D), "CS too large");

	for(size_t d = 0; d < D; ++d) {
		for(size_t n = 0; n < N; ++n) {
			while(measurements.size() < d*N*CS*R*R + (n+1)*CS*R*R) {
				std::vector<size_t> pos;
				for (size_t i = 0; i < d; ++i) {
					pos.emplace_back(dist(rnd));
				}
				pos.emplace_back(n);
				for (size_t i = d+1; i < D; ++i) {
					pos.emplace_back(dist(rnd));
				}
				
				if(!misc::contains(measurements.positions, pos)) {
					measurements.add(pos, 0.0);
				}
			}
		}
	}
	
	measurements.measure([](const std::vector<size_t>& _x){ return examples::completion::inverse_index_ratio(_x);});

	
	SinglePointMeasurementSet ctrSet = SinglePointMeasurementSet::random(D*N*CS*R*R, std::vector<size_t>(D, N), [](const std::vector<size_t>& _x){ return examples::completion::inverse_index_ratio(_x);});
	value_t ctrNorm = 0.0;
	for(size_t i = 0; i < ctrSet.size(); ++i) {
		ctrNorm += misc::sqr(ctrSet.measuredValues[i]);
	}
	ctrNorm = std::sqrt(ctrNorm);
	
	TTTensor X = TTTensor::ones(std::vector<size_t>(D, N));
	
	PerformanceData perfData([&](const TTTensor& _x){
		value_t ctrValue = 0.0;
		for(size_t i = 0; i < ctrSet.size(); ++i) {
			ctrNorm += misc::sqr(ctrSet.measuredValues[i] - _x[ctrSet.positions[i]]);
		}
		return std::sqrt(ctrValue)/ctrNorm;
	}, true, false);
	
	ADF(X, measurements, std::vector<size_t>(D-1, R), perfData);
	
	value_t ctrValue = 0.0;
	for(size_t i = 0; i < ctrSet.size(); ++i) {
		ctrNorm += misc::sqr(ctrSet.measuredValues[i] - X[ctrSet.positions[i]]);
	}
	ctrValue = std::sqrt(ctrValue)/ctrNorm;
	
	MTEST(ctrValue < 2e-2, ctrValue);

	
	perfData.reset();
	X = TTTensor::ones(std::vector<size_t>(D, N));
	
	ADF(X, RankOneMeasurementSet(measurements, X.dimensions), std::vector<size_t>(D-1, R), perfData);
	
	ctrValue = 0.0;
	for(size_t i = 0; i < ctrSet.size(); ++i) {
		ctrNorm += misc::sqr(ctrSet.measuredValues[i] - X[ctrSet.positions[i]]);
	}
	ctrValue = std::sqrt(ctrValue)/ctrNorm;
	
	MTEST(ctrValue < 2e-2, ctrValue);
});


static misc::UnitTest alg_adf_rnd("Algorithm", "adf_random_low_rank", [](){
	const size_t D = 6;
	const size_t N = 5;
	const size_t R = 3;
	const size_t CS = 10;

	TTTensor trueSolution = TTTensor::random(std::vector<size_t>(D, N), std::vector<size_t>(D-1, R));

	SinglePointMeasurementSet measurements = SinglePointMeasurementSet::random(D*N*CS*R*R, std::vector<size_t>(D, N));
	measurements.measure(trueSolution);
	
	bool test = true;
	for(size_t m = 0; m < measurements.size(); ++m) {
		test = test && misc::approx_equal(measurements.measuredValues[m], trueSolution[measurements.positions[m]], 1e-12);
	}
	TEST(test);
	
	ADFVariant ourADF(500, 1e-6, 0.999);
	
	TTTensor X = TTTensor::ones(std::vector<size_t>(D, N));
	PerformanceData perfData([&](const TTTensor& _x) {return frob_norm(_x - trueSolution)/frob_norm(trueSolution);}, true, false);
	
	ourADF(X, measurements, std::vector<size_t>(D-1, R), perfData);
	
	MTEST(frob_norm(X - trueSolution)/frob_norm(trueSolution) < 1e-3, frob_norm(X - trueSolution)/frob_norm(trueSolution));
	
	
	
	X = TTTensor::ones(std::vector<size_t>(D, N));
	perfData.reset();
	
	ourADF(X, RankOneMeasurementSet(measurements, X.dimensions), std::vector<size_t>(D-1, R), perfData);
	
	MTEST(frob_norm(X - trueSolution)/frob_norm(trueSolution) < 1e-3, frob_norm(X - trueSolution)/frob_norm(trueSolution));
});
