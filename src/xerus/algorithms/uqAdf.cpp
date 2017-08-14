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
 * @brief Implementation of the ADF variants. 
 */

#include <xerus/algorithms/uqAdf.h>

#include <xerus/misc/basicArraySupport.h>
#include <xerus/misc/simpleNumerics.h>
#include <xerus/misc/internal.h>

#include <boost/math/special_functions/hermite.hpp>
#include <boost/math/special_functions/legendre.hpp>

#ifdef _OPENMP
	#include <omp.h>
#endif

namespace xerus {
    
	Tensor randVar_to_position(const double _v, const size_t _polyDegree) {
// 		const std::vector<xerus::misc::Polynomial> stochasticBasis = xerus::misc::Polynomial::build_orthogonal_base(_polyDegree, [](const double){return 1.0;}, -1., 1.);
		
		Tensor p({_polyDegree});
		for (unsigned i = 0; i < _polyDegree; ++i) {
// 			p[i] = stochasticBasis[i](_v);
			p[i] = boost::math::hermite(i, _v/std::sqrt(2))/std::pow(2.0, i/2.0);
// 			p[i] = boost::math::legendre_p(i, _v);
// 			p[i] = boost::math::legendre_q(i, _v);
		}
		
		return p;
	}
	
    class InternalSolver {
        const size_t N;
        const size_t d;
		
		const double solutionsNorm;
        
        const std::vector<std::vector<Tensor>> positions;
        const std::vector<Tensor>& solutions;
        
        TTTensor& x;
        
		std::vector<std::vector<Tensor>> rightStack;  // From corePosition 1 to d-1
		std::vector<std::vector<Tensor>> leftIsStack;
		std::vector<std::vector<Tensor>> leftOughtStack;
		
        
        
    public:
        static std::vector<std::vector<Tensor>> create_positions(const TTTensor& _x, const std::vector<std::vector<double>>& _randomVariables) {
            std::vector<std::vector<Tensor>> positions(_x.degree());
            
            for(size_t corePosition = 1; corePosition < _x.degree(); ++corePosition) {
                positions[corePosition].reserve(_randomVariables.size());
                for(size_t j = 0; j < _randomVariables.size(); ++j) {
                    positions[corePosition].push_back(randVar_to_position(_randomVariables[j][corePosition-1], _x.dimensions[corePosition]));
                }
            }
            
            return positions;
        }
        
        static double calc_solutions_norm(const std::vector<Tensor>& _solutions) {
			double norm = 0;
			for(const auto& s : _solutions) {
				norm += misc::sqr(frob_norm(s));
			}
			
			return std::sqrt(norm);
		}
        
        
        InternalSolver(TTTensor& _x, const std::vector<std::vector<double>>& _randomVariables, const std::vector<Tensor>& _solutions) : 
            N(_randomVariables.size()), 
            d(_x.degree()),
            solutionsNorm(calc_solutions_norm(_solutions)),
            positions(create_positions(_x, _randomVariables)),
            solutions(_solutions),
            x(_x),
            rightStack(d, std::vector<Tensor>(N)),
            leftIsStack(d, std::vector<Tensor>(N)), 
            leftOughtStack(d, std::vector<Tensor>(N))
            {
                REQUIRE(_randomVariables.size() == _solutions.size(), "ERROR");
        }
        
        
        void calc_left_stack(const size_t _corePosition) {
            REQUIRE(_corePosition+1 < d, "Invalid corePosition");
            
			if(_corePosition == 0) {
				Tensor shuffledX = x.get_component(0);
				shuffledX.reinterpret_dimensions({x.dimensions[0], x.rank(0)});
				
				#pragma omp parallel for 
				for(size_t j = 0; j < N; ++j) {
                    // NOTE: leftIsStack[0] is always an identity
					contract(leftOughtStack[_corePosition][j], solutions[j], shuffledX, 1);
				}
				
			} else { // _corePosition > 0
				const Tensor shuffledX = reshuffle(x.get_component(_corePosition), {1, 0, 2});
				Tensor measCmp, tmp;
				#pragma omp parallel for  firstprivate(measCmp, tmp)
				for(size_t j = 0; j < N; ++j) {
					contract(measCmp, positions[_corePosition][j], shuffledX, 1);
					
					if(_corePosition > 1) {
						contract(tmp, measCmp, true, leftIsStack[_corePosition-1][j], false,  1);
						contract(leftIsStack[_corePosition][j], tmp, measCmp, 1);
					} else { // _corePosition == 1
						contract(leftIsStack[_corePosition][j], measCmp, true, measCmp, false, 1);
					}
					
					contract(leftOughtStack[_corePosition][j], leftOughtStack[_corePosition-1][j], measCmp, 1);
				}
			}
		}
		
        
        void calc_right_stack(const size_t _corePosition) {
            REQUIRE(_corePosition > 0 && _corePosition < d, "Invalid corePosition");
            Tensor shuffledX = reshuffle(x.get_component(_corePosition), {1, 0, 2});
            
            if(_corePosition < d-1) {
                Tensor tmp;
				#pragma omp parallel for  firstprivate(tmp)
                for(size_t j = 0; j < N; ++j) {
                    contract(tmp, positions[_corePosition][j], shuffledX, 1);
                    contract(rightStack[_corePosition][j], tmp, rightStack[_corePosition+1][j], 1);
                }
            } else { // _corePosition == d-1
                shuffledX.reinterpret_dimensions({shuffledX.dimensions[0], shuffledX.dimensions[1]}); // Remove dangling 1-mode
				#pragma omp parallel for 
                for(size_t j = 0; j < N; ++j) {
                    contract(rightStack[_corePosition][j], positions[_corePosition][j], shuffledX, 1);
                }
            }
        }
        
        
        Tensor calculate_delta(const size_t _corePosition) const {
			Tensor delta(x.get_component(_corePosition).dimensions);
			Tensor dyadComp, tmp;
			
			if(_corePosition > 0) {
				const Tensor shuffledX = reshuffle(x.get_component(_corePosition), {1, 0, 2});
				
				#pragma omp parallel for  firstprivate(dyadComp, tmp)
				for(size_t j = 0; j < N; ++j) {
					// Calculate common "dyadic part"
					Tensor dyadicPart;
					if(_corePosition < d-1) {
						contract(dyadicPart, positions[_corePosition][j], rightStack[_corePosition+1][j], 0);
					} else {
						dyadicPart = positions[_corePosition][j];
						dyadicPart.reinterpret_dimensions({dyadicPart.dimensions[0], 1}); // Add dangling 1-mode
					}
					
					
					// Calculate "is"
					Tensor isPart;
					contract(isPart, positions[_corePosition][j], shuffledX, 1);
                    
                    if(_corePosition < d-1) {
						contract(isPart, isPart, rightStack[_corePosition+1][j], 1);
                    } else {
						isPart.reinterpret_dimensions({isPart.dimensions[0]});
                    }
                    
                    if(_corePosition > 1) { // NOTE: For _corePosition == 1 leftIsStack is the identity
						contract(isPart, leftIsStack[_corePosition-1][j], isPart, 1);
                    }
                    
                    
					// Combine with ought part
					contract(dyadComp, isPart - leftOughtStack[_corePosition-1][j], dyadicPart, 0);
					
					#pragma omp critical
					{ delta += dyadComp; }
				}
			} else { // _corePosition == 0
				Tensor shuffledX = x.get_component(0);
				shuffledX.reinterpret_dimensions({shuffledX.dimensions[1], shuffledX.dimensions[2]});
				
				#pragma omp parallel for  firstprivate(dyadComp, tmp)
				for(size_t j = 0; j < N; ++j) {
					contract(dyadComp, shuffledX, rightStack[_corePosition+1][j], 1);
					contract(dyadComp, dyadComp - solutions[j], rightStack[_corePosition+1][j], 0);
					dyadComp.reinterpret_dimensions({1, dyadComp.dimensions[0], dyadComp.dimensions[1]});
					
					#pragma omp critical
					{ delta += dyadComp; }
				}
			}
            
            return delta;
        }
        
        
        double calculate_norm_A_projGrad(const Tensor& _delta, const size_t _corePosition) const {
            double norm = 0.0;
			Tensor tmp;
			
            if(_corePosition == 0) {
				#pragma omp parallel for firstprivate(tmp) reduction(+:norm)
                for(size_t j = 0; j < N; ++j) {
                    contract(tmp, _delta, rightStack[1][j], 1);
					const double normPart = misc::sqr(frob_norm(tmp));
					norm += normPart;
                }
            } else { // _corePosition > 0
                Tensor shuffledDelta = reshuffle(_delta, {1, 0, 2});
				if(_corePosition == d-1) {
					shuffledDelta.reinterpret_dimensions({shuffledDelta.dimensions[0], shuffledDelta.dimensions[1]}); // Remove dangling 1-mode
				}
                
				Tensor rightPart;
				#pragma omp parallel for  firstprivate(tmp, rightPart) reduction(+:norm)
				for(size_t j = 0; j < N; ++j) {
					// Current node
					contract(tmp, positions[_corePosition][j], shuffledDelta, 1);
					
					if(_corePosition < d-1) {
						contract(rightPart, tmp, rightStack[_corePosition+1][j], 1);
					} else {
						rightPart = tmp;
					}
					
					if(_corePosition > 1) {
						contract(tmp, rightPart, leftIsStack[_corePosition-1][j], 1);
						contract(tmp, tmp, rightPart, 1);
					} else { // NOTE: For _corePosition == 1 leftIsStack is the identity
						contract(tmp, rightPart, rightPart, 1);
					}
					
					REQUIRE(tmp.size == 1, "IE");
					norm += tmp[0];
				}
            }
            
            return std::sqrt(norm);
        }
        
        
        double calc_residual_norm(const size_t _corePosition) const {
			REQUIRE(_corePosition == 0, "Invalid corePosition");

			double norm = 0.0;
			
			Tensor tmp;
			for(size_t j = 0; j < N; ++j) {
				contract(tmp, x.get_component(0), rightStack[1][j], 1);
				tmp.reinterpret_dimensions({x.dimensions[0]});
				tmp -= solutions[j];
				norm += misc::sqr(frob_norm(tmp));
			}
			
			return std::sqrt(norm);
		}
		
        
        void solve() {
			std::vector<double> residuals(10, std::numeric_limits<double>::max());
			const size_t maxIterations = 100000;
			
			for(size_t iteration = 0; maxIterations == 0 || iteration < maxIterations; ++iteration) {
				x.move_core(0, true);
				
				// Rebuild right stack
				for(size_t corePosition = d-1; corePosition > 0; --corePosition) {
					calc_right_stack(corePosition);
				}
				
				
				for(size_t corePosition = 0; corePosition < x.degree(); ++corePosition) {
					if(corePosition == 0) {
						residuals.push_back(calc_residual_norm(0)/solutionsNorm);
						
						if(residuals.back()/residuals[residuals.size()-10] > 0.999) {
                            LOG(greee, residuals.back() << " / " << residuals[residuals.size()-10] << " = " << residuals.back()/residuals[residuals.size()-10]);
							LOG(ADF, "Residual decrease from " << std::scientific << residuals[10] << " to " << std::scientific << residuals.back() << " in " << residuals.size()-10 << " iterations.");
							return; // We are done!
						}
					}
					
					const auto delta = calculate_delta(corePosition);
					const auto normAProjGrad = calculate_norm_A_projGrad(delta, corePosition);
					const value_t PyR = misc::sqr(frob_norm(delta));
					
					// Actual update
					x.component(corePosition) -= (PyR/misc::sqr(normAProjGrad))*delta;
					
					// If we have not yet reached the end of the sweep we need to take care of the core and update our stacks
					if(corePosition+1 < d) {
						x.move_core(corePosition+1, true);
						calc_left_stack(corePosition);
					}
				}
			}
        }
    };
    
    
    
    void uq_adf(TTTensor& _x, const std::vector<std::vector<double>>& _randomVariables, const std::vector<Tensor>& _solutions) {
		LOG(ADF, "Start UQ ADF");
        InternalSolver solver(_x, _randomVariables, _solutions);
        return solver.solve();
    }
    
    
    TTTensor uq_adf(const UQMeasurementSet& _measurments, const TTTensor& _guess) {
		REQUIRE(_measurments.randomVectors.size() == _measurments.solutions.size(), "Invalid measurments");
		REQUIRE(_measurments.initialRandomVectors.size() == _measurments.initialSolutions.size(), "Invalid initial measurments");
		
		if(_measurments.initialRandomVectors.size() > 0) {
			LOG(UQ_ADF, "Init");
			TTTensor x(_guess.dimensions);
			TTTensor newX(x.dimensions);
			
			std::vector<std::vector<double>> randomVectors = _measurments.randomVectors;
			std::vector<Tensor> solutions = _measurments.solutions;
			
			// Calc mean
			Tensor mean({x.dimensions[0]});
			for(const auto& sol : solutions) {
				mean += sol;
			}
			mean /= double(solutions.size());
			
			TTTensor baseTerm(x.dimensions);
			mean.reinterpret_dimensions({1, x.dimensions[0], 1});
			baseTerm.set_component(0, mean);
			for(size_t k = 0; k < _measurments.initialRandomVectors.size(); ++k) {
				baseTerm.set_component(k+1, Tensor::dirac({1, x.dimensions[k+1], 1}, 0));
			}
			baseTerm.assume_core_position(0);
			newX += baseTerm;
			
			mean.reinterpret_dimensions({x.dimensions[0]});
			
			// Calc linear terms
			REQUIRE(_measurments.initialRandomVectors.size() == _measurments.initialRandomVectors[0].size(), "Sizes don't match.");
			REQUIRE(_measurments.initialRandomVectors.size() == _measurments.initialSolutions.size(), "Sizes don't match.");
			REQUIRE(_measurments.initialRandomVectors.size()+1 == x.degree(), "Sizes don't match.");

			for(size_t m = 0; m < _measurments.initialRandomVectors.size(); ++m) {
				REQUIRE(_measurments.initialRandomVectors[m][m] > 0.0, "Invalid initial randVec");
				TTTensor linearTerm(x.dimensions);
				Tensor tmp = (_measurments.initialSolutions[m] - mean);
				tmp.reinterpret_dimensions({1, x.dimensions[0], 1});
				linearTerm.set_component(0, tmp);
				for(size_t k = 0; k < _measurments.initialRandomVectors.size(); ++k) {
					if(k == m) {
						linearTerm.set_component(k+1, Tensor::dirac({1, x.dimensions[k+1], 1}, 0));
					} else {
						REQUIRE(misc::hard_equal(_measurments.initialRandomVectors[m][k], 0.0), "Invalid initial randVec");
						REQUIRE(x.dimensions[k+1] >= 1, "WTF");
						linearTerm.set_component(k+1, Tensor::dirac({1, x.dimensions[k+1], 1}, 1));
					}
				}
				linearTerm.assume_core_position(0);
				newX += linearTerm;
			}
			
			// Add some noise
// 			auto noise = TTTensor::random(newX.dimensions, std::vector<size_t>(newX.degree()-1, 2));
// 			noise *= 1e-4*frob_norm(newX)/frob_norm(noise);
// 			LOG(check, frob_norm(newX) << " vs " << frob_norm(noise));
// 			newX += noise;

			
			
			// Add initial measurments. NOTE: must happen after mean is calculated
			randomVectors.insert(randomVectors.end(), _measurments.initialRandomVectors.begin(), _measurments.initialRandomVectors.end());
			solutions.insert(solutions.end(), _measurments.initialSolutions.begin(), _measurments.initialSolutions.end());
			
			x = newX;
			LOG(UQ_ADF, "Pre roundign ranks: " << x.ranks());
			x.round(0.00025);
			LOG(UQ_ADF, "Post roundign ranks: " << x.ranks());
			uq_adf(x, _measurments.randomVectors, _measurments.solutions);
			return x;
		} else {
			auto x = _guess;
			uq_adf(x, _measurments.randomVectors, _measurments.solutions);
			return x;
		}
	}
	
	
	void UQMeasurementSet::add(const std::vector<double>& _rndvec, const Tensor& _solution) {
		randomVectors.push_back(_rndvec);
		solutions.push_back(_solution);
	}
	
	void UQMeasurementSet::add_initial(const std::vector<double>& _rndvec, const Tensor& _solution) {
		initialRandomVectors.push_back(_rndvec);
		initialSolutions.push_back(_solution);
	}
	
	
	std::pair<std::vector<std::vector<double>>, std::vector<Tensor>> uq_mc(const TTTensor& _x, const size_t _N, const size_t _numSpecial) {
		std::mt19937_64 rnd;
		std::normal_distribution<double> dist(0.0, 1.0);
		
		std::vector<std::vector<double>> randomVariables;
		std::vector<Tensor> solutions;
		
		randomVariables.reserve(_N);
		solutions.reserve(_N);
		for(size_t i = 0; i < _N; ++i) {
			randomVariables.push_back(std::vector<double>(_x.degree()-1));
			Tensor p = Tensor::ones({1});
			for(size_t k = _x.degree()-1; k > 0; --k) {
				randomVariables[i][k-1] = (k <= _numSpecial?0.3:1.0)*dist(rnd);
				contract(p, _x.get_component(k), p, 1);
				contract(p, p, randVar_to_position(randomVariables[i][k-1], _x.dimensions[k]), 1);
			}
			contract(p, _x.get_component(0), p, 1);
			p.reinterpret_dimensions({_x.dimensions[0]});
			solutions.push_back(p);
		}
		
		return std::make_pair(randomVariables, solutions);
	}
	
	
	Tensor uq_avg(const TTTensor& _x, const size_t _N, const size_t _numSpecial) {
		Tensor realAvg({_x.dimensions[0]});
		
		#pragma omp parallel
		{
			std::mt19937_64 rnd;
			std::normal_distribution<double> dist(0.0, 1.0);
			Tensor avg({_x.dimensions[0]});
			
			#pragma omp parallel for 
			for(size_t i = 0; i < _N; ++i) {
				Tensor p = Tensor::ones({1});
				for(size_t k = _x.degree()-1; k > 0; --k) {
					contract(p, _x.get_component(k), p, 1);
					contract(p, p, randVar_to_position((k <= _numSpecial?0.3:1.0)*dist(rnd), _x.dimensions[k]), 1);
				}
				contract(p, _x.get_component(0), p, 1);
				p.reinterpret_dimensions({_x.dimensions[0]});
				avg += p;
			}
			
			#pragma omp critical
			{ realAvg += avg; }
		}
		
		return realAvg/double(_N);
	}
	
} // namespace xerus
