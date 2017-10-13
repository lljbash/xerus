// Xerus - A General Purpose Tensor Library
// Copyright (C) 2014-2017 Benjamin Huber and Sebastian Wolf. 
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

/**
 * @file
 * @brief Implementations of the utilities for uncertainity quantification.
 */

#include <xerus/applications/uq.h>

#include <xerus/misc/math.h>
#include <xerus/misc/basicArraySupport.h>
#include <xerus/misc/internal.h>

#include <boost/math/special_functions/hermite.hpp>
#include <boost/math/special_functions/legendre.hpp>

#ifdef _OPENMP
	#include <omp.h>
#endif

namespace xerus { namespace uq {
    
	Tensor hermite_position(const double _v, const size_t _polyDegree) {
		Tensor p({_polyDegree});
		for (unsigned i = 0; i < _polyDegree; ++i) {
			p[i] = boost::math::hermite(i, _v/std::sqrt(2))/std::pow(2.0, i/2.0);
		}
		return p;
	}
	
	
	Tensor legendre_position(const double _v, const size_t _polyDegree) {
		Tensor p({_polyDegree});
		for (unsigned i = 0; i < _polyDegree; ++i) {
			p[i] = boost::math::legendre_p(i, _v);
		}
		return p;
	}
	
	
    void UQMeasurementSet::add(const std::vector<double>& _rndvec, const Tensor& _solution) {
		randomVectors.push_back(_rndvec);
		solutions.push_back(_solution);
	}
	
	
	void UQMeasurementSet::add_initial(const std::vector<double>& _rndvec, const Tensor& _solution) {
		initialRandomVectors.push_back(_rndvec);
		initialSolutions.push_back(_solution);
	}
	
	
	void UQMeasurementSet::clear() {
		randomVectors.clear();
		solutions.clear();
		initialRandomVectors.clear();
		initialSolutions.clear();
	}
	
    
    TTTensor initial_guess(const UQMeasurementSet& _measurments, const TTTensor& _guess) {
		REQUIRE(_measurments.randomVectors.size() == _measurments.solutions.size(), "Invalid measurments");
		REQUIRE(_measurments.initialRandomVectors.size() == _measurments.initialSolutions.size(), "Invalid initial measurments");
		
        LOG(UQ, "Init");
		if(_measurments.initialRandomVectors.size() > 0) {
            TTTensor x(_guess.dimensions);
			
			// Calc mean
			Tensor mean({x.dimensions[0]});
			for(const auto& sol : _measurments.solutions) {
				mean += sol;
			}
			mean /= double(_measurments.solutions.size());
			
            // Set mean
			mean.reinterpret_dimensions({1, x.dimensions[0], 1});
			x.set_component(0, mean);
			for(size_t k = 1; k < x.degree(); ++k) {
				x.set_component(k, Tensor::dirac({1, x.dimensions[k], 1}, 0));
			}
			x.assume_core_position(0);
			
			mean.reinterpret_dimensions({x.dimensions[0]});

			// Calc linear terms
            std::set<size_t> usedParams;
			for(size_t m = 0; m < _measurments.initialRandomVectors.size(); ++m) {
                const auto& rndVec = _measurments.initialRandomVectors[m];
                const auto& sol = _measurments.initialSolutions[m];
                
                REQUIRE(rndVec.size()+1 == x.degree(), "Invalid random vector");
                
                // Find parameter number
                size_t p = x.degree();
                bool skip = false;
                for(size_t i = 0; i < rndVec.size(); ++i) {
                    if(std::abs(rndVec[i]) > 0.0) {
                        if(misc::contains(usedParams, i)) {
                            LOG(info, "Skipping douplicate parameter " << i);
                            skip = true;
                            continue; 
                        }
                        REQUIRE(p == x.degree(), "Parameters contains several non-zero entries: " << rndVec);
                        REQUIRE(!misc::contains(usedParams, i), "Parameters " << i << " appears twice!" << _measurments.initialRandomVectors);
                        usedParams.emplace(i);
                        p = i;
                    }
                }
                if(skip) { continue; }
                REQUIRE(p != x.degree(), "Parameters contains no non-zero entry: " << rndVec);
                
				TTTensor linearTerm(x.dimensions);
				Tensor tmp = (sol - mean)/rndVec[p]; // TODO
				tmp.reinterpret_dimensions({1, x.dimensions[0], 1});
				linearTerm.set_component(0, tmp);
				for(size_t k = 1; k < x.degree(); ++k) {
					if(k == p+1) {
						linearTerm.set_component(k, Tensor::dirac({1, x.dimensions[k], 1}, 1));
					} else {
						REQUIRE(misc::hard_equal(rndVec[k-1], 0.0), "Invalid initial randVec.");
						linearTerm.set_component(k, Tensor::dirac({1, x.dimensions[k], 1}, 0));
					}
				}
				linearTerm.assume_core_position(0);
				x += linearTerm;
			}
			
			LOG(UQ_Inital_Guess, "Found linear terms for " << usedParams);
			
			x.round(1e-4);
			LOG(UQ_Inital_Guess, "Post roundign ranks: " << x.ranks());
			return x;
		} else {
			LOG(info, "UQ initial guess: Nothing done, because no measurments were provided.");
			return _guess;
		}
	}
	
	
	Tensor mc_average(const TTTensor& _x, const PolynomBasis _basisType, const size_t _N) {
		Tensor realAvg({_x.dimensions[0]});
		
		#pragma omp parallel
		{
			std::mt19937_64 rnd;
			Tensor avg({_x.dimensions[0]});
			const Tensor one = Tensor::ones({1});
			
			if(_basisType == PolynomBasis::Hermite) {
				std::normal_distribution<double> dist(0.0, 1.0);
				
				#pragma omp parallel for 
				for(size_t i = 0; i < _N; ++i) {
					Tensor p = one;
					for(size_t k = _x.degree()-1; k > 0; --k) {
						contract(p, _x.get_component(k), p, 1);
						contract(p, p, hermite_position(dist(rnd), _x.dimensions[k]), 1);
					}
					contract(p, _x.get_component(0), p, 1);
					p.reinterpret_dimensions({_x.dimensions[0]});
					avg += p;
				}
			} else {
				std::uniform_real_distribution<double> dist(-1.0, 1.0);
				
				#pragma omp parallel for 
				for(size_t i = 0; i < _N; ++i) {
					Tensor p = one;
					for(size_t k = _x.degree()-1; k > 0; --k) {
						contract(p, _x.get_component(k), p, 1);
						contract(p, p, legendre_position(dist(rnd), _x.dimensions[k]), 1);
					}
					contract(p, _x.get_component(0), p, 1);
					p.reinterpret_dimensions({_x.dimensions[0]});
					avg += p;
				}
			}
			
			#pragma omp critical
			{ realAvg += avg; }
		}
		
		return realAvg/double(_N);
	}
	
	
	
	Tensor average(const TTTensor& _x) {
		TTTensor x = _x;
		
		while(x.degree() > 1) {
			x.fix_mode(1, 0);
		}
		
		return Tensor(x);
	}
	
} // namespace uq 
} // namespace xerus

