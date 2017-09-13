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

#include <xerus/applications/uqAdf.h>

#include <xerus/misc/basicArraySupport.h>
#include <xerus/misc/math.h>
#include <xerus/misc/internal.h>

#include <boost/math/special_functions/hermite.hpp>
#include <boost/math/special_functions/legendre.hpp>

#ifdef _OPENMP
	#include <omp.h>
#endif

namespace xerus { namespace uq {
	
	namespace impl_ra_adf {
    class InternalSolver {
        const size_t N;
        const size_t P;
		const size_t d;
		const double rankEps = 1e-6;
		
		///@brief [setId]
		std::vector<double> solutionsNorms;
        
        ///@brief [setId][corePosition][i]
        std::vector<std::vector<std::vector<Tensor>>> positionSets;
		///@brief [setId][i]]
		std::vector<std::vector<Tensor>> solutionSets;
        
        TTTensor& x;
		
		///@brief leftRank x ext x P x rightRank
		Tensor currComp;
		
		///@brief [setId][corePosition][i]
		std::vector<std::vector<std::vector<Tensor>>> rightStacks;  // From corePosition 1 to d-1
		///@brief [setId][corePosition][i]
		std::vector<std::vector<std::vector<Tensor>>> leftIsStacks;
		///@brief [setId][corePosition][i]
		std::vector<std::vector<std::vector<Tensor>>> leftOughtStacks;
		
        
        
    public:
		void calc_solution_norms() {
			for(size_t setId = 0; setId < P; ++setId) {
				LOG(info, "Set size " << setId <<": " << solutionSets[setId].size());
				double norm = 0;
				for(const auto& s : solutionSets[setId]) {
					norm += misc::sqr(frob_norm(s));
				}
				
				solutionsNorms[setId] = std::sqrt(norm);
			}
		}
        
        
        InternalSolver(TTTensor& _x, const std::vector<std::vector<double>>& _randomVariables, const std::vector<Tensor>& _solutions, const size_t _P) : 
            N(_randomVariables.size()),
            P(_P),
            d(_x.degree()),
            solutionsNorms(_P),
            positionSets(_P, std::vector<std::vector<Tensor>>(_x.degree())),
            solutionSets(P),
            x(_x),
            rightStacks(_P, std::vector<std::vector<Tensor>>(_x.degree(), std::vector<Tensor>(N))),
            leftIsStacks(_P, std::vector<std::vector<Tensor>>(_x.degree(), std::vector<Tensor>(N))), 
            leftOughtStacks(_P, std::vector<std::vector<Tensor>>(_x.degree(), std::vector<Tensor>(N)))
            {
				REQUIRE(P > 0, "P must be > 0");
				REQUIRE(_randomVariables.size() == _solutions.size(), "There must be the same amount of randomVectors and solutions.");
				std::uniform_int_distribution<size_t> setDist(0, P-1);
				
				for(size_t j = 0; j < N; ++j) {
					const size_t setId = setDist(misc::randomEngine);
					for(size_t corePosition = 1; corePosition < _x.degree(); ++corePosition) {
						positionSets[setId][corePosition].push_back(hermite_position(_randomVariables[j][corePosition-1], _x.dimensions[corePosition])); // TODO choose polynom basis
						solutionSets[setId].push_back(_solutions[j]);
					}
				}
				
				calc_solution_norms();
        }
        
        void move_core(size_t _from, const size_t _to) {
			const Index left, right, ext, p, r1, r2;
			Tensor U, S, V;
			while(_from < _to) { // To right
				(U(left, ext, r1), S(r1, r2), V(r2, p, right)) = SVD(currComp(left, ext, p, right), std::numeric_limits<size_t>::max(), rankEps);
				x.set_component(_from, U);
				currComp(left, ext, p, right) = S(left, r1)*V(r1, p, r2)*x.get_component(_from+1)(r2, ext, right);
				_from++;
			}
			
			while(_from > _to) { // To left
				(U(left, p, r1), S(r1, r2), V(r2, ext, right)) = SVD(currComp(left, ext, p, right), std::numeric_limits<size_t>::max(), rankEps);
				x.set_component(_from, V);
				currComp(left, ext, p, right) = x.get_component(_from-1)(left, ext, r1)*U(r1, p, r2)*S(r2, right);
				_from--;
			}
		}
        
        
        void calc_left_stacks(const size_t _corePosition) {
			REQUIRE(_corePosition+1 < d, "Invalid corePosition");
			for(size_t setId = 0; setId < P; ++setId) {
				if(_corePosition == 0) {
					Tensor shuffledX = x.get_component(0);
					shuffledX.reinterpret_dimensions({x.dimensions[0], x.rank(0)});
					
					#pragma omp parallel for 
					for(size_t j = 0; j < N; ++j) {
						// NOTE: leftIsStack[0] is always an identity
						contract(leftOughtStacks[setId][_corePosition][j], solutionSets[setId][j], shuffledX, 1);
					}
					
				} else { // _corePosition > 0
					const Tensor shuffledX = reshuffle(x.get_component(_corePosition), {1, 0, 2});
					Tensor measCmp, tmp;
					#pragma omp parallel for  firstprivate(measCmp, tmp)
					for(size_t j = 0; j < N; ++j) {
						contract(measCmp, positionSets[setId][_corePosition][j], shuffledX, 1);
						
						if(_corePosition > 1) {
							contract(tmp, measCmp, true, leftIsStacks[setId][_corePosition-1][j], false,  1);
							contract(leftIsStacks[setId][_corePosition][j], tmp, measCmp, 1);
						} else { // _corePosition == 1
							contract(leftIsStacks[setId][_corePosition][j], measCmp, true, measCmp, false, 1);
						}
						
						contract(leftOughtStacks[setId][_corePosition][j], leftOughtStacks[setId][_corePosition-1][j], measCmp, 1);
					}
				}
			}
		}
		
        
        void calc_right_stack(const size_t _corePosition) {
            REQUIRE(_corePosition > 0 && _corePosition < d, "Invalid corePosition");
			for(size_t setId = 0; setId < P; ++setId) {
				Tensor shuffledX = reshuffle(x.get_component(_corePosition), {1, 0, 2});
				
				if(_corePosition < d-1) {
					Tensor tmp;
					#pragma omp parallel for  firstprivate(tmp)
					for(size_t j = 0; j < N; ++j) {
						contract(tmp, positionSets[setId][_corePosition][j], shuffledX, 1);
						contract(rightStacks[setId][_corePosition][j], tmp, rightStacks[setId][_corePosition+1][j], 1);
					}
				} else { // _corePosition == d-1
					shuffledX.reinterpret_dimensions({shuffledX.dimensions[0], shuffledX.dimensions[1]}); // Remove dangling 1-mode
					#pragma omp parallel for 
					for(size_t j = 0; j < N; ++j) {
						contract(rightStacks[setId][_corePosition][j], positionSets[setId][_corePosition][j], shuffledX, 1);
					}
				}
			}
        }
        
        
        Tensor calculate_delta(const size_t _corePosition, const size_t _setId) const {
			Tensor delta(x.get_component(_corePosition).dimensions);
			Tensor dyadComp, tmp;
			
			if(_corePosition > 0) {
				Tensor shuffledX = currComp;
				shuffledX.fix_mode(2, _setId);
				shuffledX = reshuffle(shuffledX, {1, 0, 2});
				
				#pragma omp parallel for  firstprivate(dyadComp, tmp)
				for(size_t j = 0; j < N; ++j) {
					// Calculate common "dyadic part"
					Tensor dyadicPart;
					if(_corePosition < d-1) {
						contract(dyadicPart, positionSets[_setId][_corePosition][j], rightStacks[_setId][_corePosition+1][j], 0);
					} else {
						dyadicPart = positionSets[_setId][_corePosition][j];
						dyadicPart.reinterpret_dimensions({dyadicPart.dimensions[0], 1}); // Add dangling 1-mode
					}
					
					
					// Calculate "is"
					Tensor isPart;
					contract(isPart, positionSets[_setId][_corePosition][j], shuffledX, 1);
                    
                    if(_corePosition < d-1) {
						contract(isPart, isPart, rightStacks[_setId][_corePosition+1][j], 1);
                    } else {
						isPart.reinterpret_dimensions({isPart.dimensions[0]});
                    }
                    
                    if(_corePosition > 1) { // NOTE: For _corePosition == 1 leftIsStack is the identity
						contract(isPart, leftIsStacks[_setId][_corePosition-1][j], isPart, 1);
                    }
                    
                    
					// Combine with ought part
					contract(dyadComp, isPart - leftOughtStacks[_setId][_corePosition-1][j], dyadicPart, 0);
					
					#pragma omp critical
					{ delta += dyadComp; }
				}
			} else { // _corePosition == 0
				Tensor shuffledX = currComp;
				shuffledX.fix_mode(2, _setId);
				shuffledX.reinterpret_dimensions({shuffledX.dimensions[1], shuffledX.dimensions[2]});
				
				#pragma omp parallel for  firstprivate(dyadComp, tmp)
				for(size_t j = 0; j < N; ++j) {
					contract(dyadComp, shuffledX, rightStacks[_setId][_corePosition+1][j], 1);
					contract(dyadComp, dyadComp - solutionSets[_setId][j], rightStacks[_setId][_corePosition+1][j], 0);
					dyadComp.reinterpret_dimensions({1, dyadComp.dimensions[0], dyadComp.dimensions[1]});
					
					#pragma omp critical
					{ delta += dyadComp; }
				}
			}
            
            return delta;
        }
        
        
        double calculate_norm_A_projGrad(const Tensor& _delta, const size_t _corePosition, const size_t _setId) const {
            double norm = 0.0;
			Tensor tmp;
			
            if(_corePosition == 0) {
				#pragma omp parallel for firstprivate(tmp) reduction(+:norm)
                for(size_t j = 0; j < N; ++j) {
					contract(tmp, _delta, rightStacks[_setId][1][j], 1);
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
					contract(tmp, positionSets[_setId][_corePosition][j], shuffledDelta, 1);
					
					if(_corePosition < d-1) {
						contract(rightPart, tmp, rightStacks[_setId][_corePosition+1][j], 1);
					} else {
						rightPart = tmp;
					}
					
					if(_corePosition > 1) {
						contract(tmp, rightPart, leftIsStacks[_setId][_corePosition-1][j], 1);
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
			for(size_t setId = 0; setId < P; ++setId) {
				Tensor tmp;
				for(size_t j = 0; j < N; ++j) {
					auto cmp = currComp;
					cmp.fix_mode(2, setId);
					contract(tmp, cmp, rightStacks[setId][1][j], 1);
					tmp.reinterpret_dimensions({x.dimensions[0]});
					tmp -= solutionSets[setId][j];
					norm += misc::sqr(frob_norm(tmp));
				}
			}
			
			return std::sqrt(norm);
		}
		
        static double l2_norm(const std::vector<double>& _vec) {
			double norm = 0.0;
			for(const auto val : _vec) {
				norm += misc::sqr(val);
			}
			return std::sqrt(norm);
		}
        
        void solve() {
			std::vector<double> residuals(10, std::numeric_limits<double>::max());
			const size_t maxIterations = 100000;
			LOG(bla, "Move: " << x.dimensions << ", " << x.ranks());
			x.move_core(d-1);
			currComp = Tensor({x.rank(d-2), x.dimensions[d-1], P, 1}, Tensor::Representation::Dense);
			for(size_t setId = 0; setId < P; ++setId) {
				Tensor add;
				contract(add, x.get_component(d-1), Tensor::dirac({1, P, 1}, setId), 1);
				currComp += add;
			}
			
			for(size_t iteration = 0; maxIterations == 0 || iteration < maxIterations; ++iteration) {
				move_core(d-1, 0);
				
				// Rebuild right stack
				for(size_t corePosition = d-1; corePosition > 0; --corePosition) {
					calc_right_stack(corePosition);
				}
				
				residuals.push_back(calc_residual_norm(0)/l2_norm(solutionsNorms));
				
				LOG(bla, "Itr: " << x.dimensions << ", " << x.ranks());
				LOG(ADFx, "Residual " << std::scientific << residuals.back());
				
				if(residuals.back()/residuals[residuals.size()-10] > 0.999) {
					LOG(greee, residuals.back() << " / " << residuals[residuals.size()-10] << " = " << residuals.back()/residuals[residuals.size()-10]);
					LOG(ADF, "Residual decrease from " << std::scientific << residuals[10] << " to " << std::scientific << residuals.back() << " in " << residuals.size()-10 << " iterations.");
					return; // We are done!
				}
				
				for(size_t corePosition = 0; corePosition < d; ++corePosition) {
					for(size_t setId = 0; setId < P; ++setId) {
						const auto delta = calculate_delta(corePosition, setId);
						const auto normAProjGrad = calculate_norm_A_projGrad(delta, corePosition, setId);
						const value_t PyR = misc::sqr(frob_norm(delta));
						
						// Actual update
						const Index left, right, ext, p;
						currComp(left, ext, p, right)  = currComp(left, ext, p, right)-((PyR/misc::sqr(normAProjGrad))*delta)(left, ext, right)*Tensor::dirac({P}, setId)(p);
					}
					
					// If we have not yet reached the end of the sweep we need to take care of the core and update our stacks
					if(corePosition+1 < d) {
						move_core(corePosition, corePosition+1);
						calc_left_stacks(corePosition);
					}
				}
			}
        }
    };
	
	}
    
    
    
    void uq_ra_adf(TTTensor& _x, const std::vector<std::vector<double>>& _randomVariables, const std::vector<Tensor>& _solutions) {
		LOG(ADF, "Start UQ ADF");
		impl_ra_adf::InternalSolver solver(_x, _randomVariables, _solutions, 1);
        return solver.solve();
    }
    
    
    TTTensor uq_ra_adf(const UQMeasurementSet& _measurments, const TTTensor& _guess) {
		REQUIRE(_measurments.randomVectors.size() == _measurments.solutions.size(), "Invalid measurments");
		REQUIRE(_measurments.initialRandomVectors.size() == _measurments.initialSolutions.size(), "Invalid initial measurments");
		
        TTTensor x = initial_guess(_measurments, _guess);
        
        std::vector<std::vector<double>> randomVectors = _measurments.randomVectors;
        std::vector<Tensor> solutions = _measurments.solutions;
        randomVectors.insert(randomVectors.end(), _measurments.initialRandomVectors.begin(), _measurments.initialRandomVectors.end());
        solutions.insert(solutions.end(), _measurments.initialSolutions.begin(), _measurments.initialSolutions.end());
        
		uq_ra_adf(x, _measurments.randomVectors, _measurments.solutions);
        return x;
	}

	
}} // namespace  uq | xerus
