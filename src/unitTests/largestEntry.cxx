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
using namespace xerus;


static misc::UnitTest alg_largestEntry("Algorithm", "LargestEntry", [](){
    //Random numbers
    std::mt19937_64 rnd = xerus::misc::randomEngine;
	std::uniform_int_distribution<size_t> dimDist(1,3);
	std::uniform_int_distribution<size_t> rankDist(1,4);
    
	const size_t D = 16;
	
	for(size_t k = 0; k < 2; ++k) {
		std::vector<size_t> stateDims;
		stateDims.push_back(dimDist(rnd));
		
		std::vector<size_t> ranks;
		
		for(size_t d = 2; d <= D; ++d) {
			stateDims.push_back(dimDist(rnd));
			ranks.push_back(rankDist(rnd));
			INTERNAL_CHECK(d == stateDims.size() && d == ranks.size()+1, "IE");
			
			TTTensor X = TTTensor::random(stateDims, ranks);
			X /= X.frob_norm();
			
			Tensor fullX(X);
			
			size_t posA = 0, posB = 0;
			for(size_t i = 1; i < fullX.size; ++i) {
				if(std::abs(fullX[i]) >= std::abs(fullX[posA])) {
					posB = posA;
					posA = i;
				}
				else if(std::abs(fullX[i]) >= std::abs(fullX[posB])) {
					posB = i;
				}
			}
			LOG(largestEntry, "Largest entries are: " << fullX[posA] << " and " << fullX[posB] << " at " << posA << " and " << posB);
			
			double alpha = std::abs(fullX[posB]/fullX[posA]);
			double Xn = std::abs(fullX[posA]);
			
			size_t position = find_largest_entry(X, alpha, Xn);
			
			LOG(largestEntry, "Result: " << fullX[position] << " vs " << fullX[posA] << " at positions " << position << " and " << posA);
			TEST(position == posA);
			if(position != posA) {
				LOG(omg, fullX.to_string());
			}
		}
	}
});

// UNIT_TEST(Algorithm, rankRange,
//     //Random numbers
//     std::mt19937_64 rnd;
//     rnd.seed(73);
// 	std::normal_distribution<value_t> dist (0.0, 1.0);
//     
// 	const size_t R = 5;
// 	
// 	std::vector<size_t> runtimes(R+1, 0);
// 	std::vector<size_t> interationCounts(R+1, 0);
// 	std::vector<size_t> maxRanks(R+1, 0);
// 	misc::TimeMeasure clock;
// 	
// 	size_t d = 20;
// 	
// 	for(size_t k = 0; k < 100000; ++k) {
// 		runtimes[0]++;
// 		maxRanks[0]++;
// 		interationCounts[0]++;
// 		for(size_t r = 1; r <= R; ++r) {
// 			LOG(bla, r);
// 			TTTensor X = TTTensor::random(std::vector<size_t>(d, 2), std::vector<size_t>(d-1, r), dist);
// 			size_t maxRank = 0, interationCount = 0;
// 			clock.step();
// 			size_t position = X.find_largest_entry(0.99, maxRank, interationCount);
// 			runtimes[r] += clock.get();
// 			interationCounts[r] += interationCount;
// 			maxRanks[r] += maxRank;
// 		}
// 		std::ofstream out("normalRankRange.dat");
// 		for(size_t r = 2; r <= R; ++r) {
// 			out << r << " " << runtimes[r]/runtimes[0] << " " << double(interationCounts[r])/double(interationCounts[0]) << " " << double(maxRanks[r])/double(maxRanks[0]) << std::endl;
// 		}
// 		out.close();
// 	}
// )

// UNIT_TEST(Algorithm, orderRange,
//     //Random numbers
//     std::mt19937_64 rnd;
//     rnd.seed(73);
// 	std::uniform_real_distribution<value_t> dist (0.0, 1.0);
//     
// 	const size_t D = 62;
// 	
// 	std::vector<size_t> runtimes(D+1, 0);
// 	std::vector<size_t> interationCounts(D+1, 0);
// 	std::vector<size_t> maxRanks(D+1, 0);
// 	misc::TimeMeasure clock;
// 	
// 	for(size_t k = 0; k < 100000; ++k) {
// 		runtimes[0]++;
// 		maxRanks[0]++;
// 		interationCounts[0]++;
// 		for(size_t d = 2; d <= D; ++d) {
// 			TTTensor X = TTTensor::random(std::vector<size_t>(d, 2), std::vector<size_t>(d-1, 2), dist);
// 			size_t maxRank = 0, interationCount = 0;
// 			clock.step();
// 			size_t position = X.find_largest_entry(0.99, maxRank, interationCount);
// 			runtimes[d] += clock.get();
// 			interationCounts[d] += interationCount;
// 			maxRanks[d] += maxRank;
// 		}
// 		
// 		std::ofstream out("uniOrderRange.dat");
// 		for(size_t d = 2; d <= D; ++d) {
// 			out << d << " " << runtimes[d]/runtimes[0] << " " << double(interationCounts[d])/double(interationCounts[0]) << " " << double(maxRanks[d])/double(maxRanks[0]) << std::endl;
// 		}
// 		out.close();
// 	}
// )

/*
UNIT_TEST(Algorithm, largestEntryData,
    //Random numbers
//     std::mt19937_64 rnd(rd());
// 	std::uniform_real_distribution<value_t> dist(0.0, 1.0);
	std::normal_distribution<value_t> dist(0.0, 1.0);
    
	const size_t d = 16;
	
	// 1267206426 - boeser seed fuer uniform_real_distribution(0.0, 1.0), 2^16 rank 4
	
	std::ofstream out("largestEntry.dat", std::ios::app);
	
	for(size_t k = 0; k < 10000; ++k) {
		size_t seed = rd();
		rnd.seed(seed);
		std::vector<size_t> stateDims(d,2);
		std::vector<size_t> ranks(d-1,4);
		
		TTTensor X = TTTensor::random(stateDims, ranks, dist);
// 		TTTensor X = examples::peaking_diagonals(d, 2);
		X /= X.frob_norm();
		
		Tensor fullX(X);
		
		size_t posA = 0, posB = 0;
		for(size_t i = 1; i < fullX.size; ++i) {
			if(std::abs(fullX[i]) >= std::abs(fullX[posA])) {
				posB = posA;
				posA = i;
			}
			else if(std::abs(fullX[i]) >= std::abs(fullX[posB])) {
				posB = i;
			}
		}
// 		LOG(largestEntry, "Largest entries are: " << fullX[posA] << " and " << fullX[posB] << " at " << posA << " and " << posB);
		
		double alpha = std::abs(fullX[posB]/fullX[posA]); // 0.5;
		TTTensor origX(X);
		misc::TimeMeasure clock;
		
		
		for (size_t o=0; o<=30; ++o) {
			try {
				double xfactor = o==30? 1000 : misc::pow(1.1, o)/misc::pow(1.1, 9);
				double Xn = std::abs(fullX[posA])*xfactor;
				X = origX;
				double tau = (1-alpha)*alpha*Xn*Xn/(2.0*double(d-1));
// 				LOG(largestEntry, alpha << " >> " << tau);
				
				size_t maxRank = 0;
				size_t rankSum = 0;
				clock.step();
				while(misc::sum(X.ranks()) >= d) {
					X = TTTensor::entrywise_product(X, X);
// 					LOG(largestEntry, "Before ST: " << X.ranks() << " --- " << X.frob_norm());
					X.soft_threshold(tau, true);
// 					LOG(largestEntry, "After ST: " << X.ranks() << " --- " << X.frob_norm());
					maxRank = std::max(maxRank, misc::max(X.ranks()));
					rankSum += misc::sum(X.ranks());
					
					Xn = std::max(0.0, (1-(1-alpha)*alpha/2.0))*Xn*Xn;
					double fNorm = X.frob_norm();
					Xn /= fNorm;
					X /= fNorm;
					tau = (1-alpha)*alpha*Xn*Xn/(2.0*double(d-1));
				}
				
				size_t position = 0;
				size_t factor = misc::product(stateDims);
				for(size_t c = 0; c < d; ++c) {
					factor /= stateDims[c];
					size_t maxPos = 0;
					for(size_t i = 1; i < stateDims[c]; ++i) {
						if(std::abs(X.get_component(c)[i]) > std::abs(X.get_component(c)[maxPos])) {
							maxPos = i;
						}
					}
					position += maxPos*factor;
				}
				
	// 			LOG(largestEntry, "Result: " << fullX[position] << " vs " << fullX[posA] << " at positions " << position << " and " << posA);
				std::cout << (position == posA ? '+' : '-') << std::flush;
				
				out <<  xfactor << " " << maxRank << " " << rankSum << " " << (1-std::abs(fullX[position]/fullX[posA])) << " " << clock.get() << " " << seed << std::endl;
				
	// 			alpha /= 1.1;
			} catch (const misc::generic_error &) {
				std::cout << 'F';
			}
		}
		
		std::cout << '|' << std::flush;
// 		TEST(position == posA);
// 		if(position != posA) {
// 			LOG(omg, fullX.to_string());
// 		}
	}
	out.close();
)
*/
