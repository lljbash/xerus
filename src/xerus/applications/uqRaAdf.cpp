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

#include <xerus/blockTT.h>
#include <xerus/misc/basicArraySupport.h>
#include <xerus/misc/math.h>
#include <xerus/misc/internal.h>

#ifdef _OPENMP
	#include <omp.h>
#endif

namespace xerus { namespace uq { namespace impl_uqRaAdf {
	
	class InternalSolver {
		const size_t N;
		const size_t P = 2;
		const size_t d;
		
		const PolynomBasis basisType;
		
		const double targetResidual;
		
		const size_t maxRank = 50;
		double rankEps;
		double minRankEps = 1e-7;
		const double epsDecay = 0.975;
		
		const double convergenceFactor = 0.995;
		const size_t maxIterations;
		
		double optNorm;
		double testNorm;
		std::vector<double> setNorms;
		
		const std::vector<std::vector<Tensor>> positions;
		const std::vector<Tensor>& solutions;
		
		internal::BlockTT x;
		
		double bestTestResidual = std::numeric_limits<double>::max();
		internal::BlockTT bestX;
		
		TTTensor& outX;
		
		std::vector<std::vector<Tensor>> rightStack;  // From corePosition 1 to d-1
		std::vector<std::vector<Tensor>> leftIsStack;
		std::vector<std::vector<Tensor>> leftOughtStack;
		
		std::vector<std::vector<size_t>> sets;
		std::vector<size_t> controlSet;
		
		std::vector<double> residuals = std::vector<double>(10, std::numeric_limits<double>::max());
		
		
	public:
		static std::vector<std::vector<Tensor>> create_positions(const TTTensor& _x, const PolynomBasis _basisType, const std::vector<std::vector<double>>& _randomVariables) {
			std::vector<std::vector<Tensor>> positions(_x.degree());
			
			for(size_t corePosition = 1; corePosition < _x.degree(); ++corePosition) {
				positions[corePosition].reserve(_randomVariables.size());
				for(size_t j = 0; j < _randomVariables.size(); ++j) {
					if(_basisType == PolynomBasis::Hermite) {
						positions[corePosition].push_back(hermite_position(_randomVariables[j][corePosition-1], _x.dimensions[corePosition]));
					} else {
						positions[corePosition].push_back(legendre_position(_randomVariables[j][corePosition-1], _x.dimensions[corePosition]));
					}
				}
			}
			
			return positions;
		}
		
		
		void calc_solution_norms() {
			optNorm = 0.0;
			testNorm = 0.0;
			for(size_t k = 0; k < sets.size(); ++k) {
				setNorms[k] = 0.0;
			}
			
			for(size_t j = 0; j < N; ++j) {
				const double sqrNorm = misc::sqr(frob_norm(solutions[j]));
				if(misc::contains(controlSet, j)){
					testNorm += sqrNorm;
				} else {
					optNorm += sqrNorm;
					
					for(size_t k = 0; k < sets.size(); ++k) {
						if(misc::contains(sets[k], j)) {
							setNorms[k] += sqrNorm;
						}
					}
				}
			}
			
			optNorm = std::sqrt(optNorm);
			testNorm = std::sqrt(testNorm);
			for(size_t k = 0; k < sets.size(); ++k) {
				setNorms[k] = std::sqrt(setNorms[k]);
			}
		}
		
		
		void shuffle_sets() {
			// Reset sets
			sets = std::vector<std::vector<size_t>>(P);
			controlSet.clear();
			
			std::uniform_real_distribution<double> stochDist(0.0, 1.0);
			std::uniform_int_distribution<size_t> setDist(0, P-1);
			
			for(size_t j = 0; j < N; ++j) {
				if(stochDist(misc::randomEngine) > 0.1) {
					sets[setDist(misc::randomEngine)].push_back(j);
				} else {
					controlSet.push_back(j);
				}
			}
			
			calc_solution_norms();
		}
		
		
		InternalSolver(TTTensor& _x, const std::vector<std::vector<double>>& _randomVariables, const std::vector<Tensor>& _solutions, const PolynomBasis _basisType, const size_t _maxItr, const double _targetEps, const double _initalRankEps) : 
			N(_randomVariables.size()),
			d(_x.degree()),
			basisType(_basisType),
			targetResidual(_targetEps),
			rankEps(_initalRankEps),
			maxIterations(_maxItr),
			setNorms(P),
			positions(create_positions(_x, _basisType, _randomVariables)),
			solutions(_solutions),
			x(_x, 0, P),
			outX(_x),
			rightStack(d, std::vector<Tensor>(N)),
			leftIsStack(d, std::vector<Tensor>(N)), 
			leftOughtStack(d, std::vector<Tensor>(N))
			{
				REQUIRE(_randomVariables.size() == _solutions.size(), "ERROR");
				LOG(uqADF, "Set size: " << _solutions.size());
				
				shuffle_sets();
		}
		
		
		void calc_left_stack(const size_t _position) {
			REQUIRE(_position+1 < d, "Invalid corePosition");
			
			if(_position == 0) {
				Tensor shuffledX = x.get_component(0);
				shuffledX.reinterpret_dimensions({x.dimensions[0], x.rank(0)});
				
				#pragma omp parallel for 
				for(size_t j = 0; j < N; ++j) {
					// NOTE: leftIsStack[0] is always an identity
					contract(leftOughtStack[_position][j], solutions[j], shuffledX, 1);
				}
				
			} else { // _corePosition > 0
				const Tensor shuffledX = reshuffle(x.get_component(_position), {1, 0, 2});
				Tensor measCmp, tmp;
				#pragma omp parallel for firstprivate(measCmp, tmp)
				for(size_t j = 0; j < N; ++j) {
					contract(measCmp, positions[_position][j], shuffledX, 1);
					
					if(_position > 1) {
						contract(tmp, measCmp, true, leftIsStack[_position-1][j], false,  1);
						contract(leftIsStack[_position][j], tmp, measCmp, 1);
					} else { // _corePosition == 1
						contract(leftIsStack[_position][j], measCmp, true, measCmp, false, 1);
					}
					
					contract(leftOughtStack[_position][j], leftOughtStack[_position-1][j], measCmp, 1);
				}
			}
		}
		
		
		void calc_right_stack(const size_t _position) {
			REQUIRE(_position > 0 && _position < d, "Invalid corePosition");
			Tensor shuffledX = reshuffle(x.get_component(_position), {1, 0, 2});
			
			if(_position < d-1) {
				Tensor tmp;
				#pragma omp parallel for firstprivate(tmp)
				for(size_t j = 0; j < N; ++j) {
					contract(tmp, positions[_position][j], shuffledX, 1);
					contract(rightStack[_position][j], tmp, rightStack[_position+1][j], 1);
				}
			} else { // _corePosition == d-1
				shuffledX.reinterpret_dimensions({shuffledX.dimensions[0], shuffledX.dimensions[1]}); // Remove dangling 1-mode
				#pragma omp parallel for
				for(size_t j = 0; j < N; ++j) {
					contract(rightStack[_position][j], positions[_position][j], shuffledX, 1);
				}
			}
		}
		
		
		Tensor calculate_delta(const size_t _corePosition, const size_t _setId) const {
			REQUIRE(x.corePosition == _corePosition, "IE");
			
			Tensor delta(x.get_core(_setId).dimensions);
			Tensor dyadComp, tmp;
			
			if(_corePosition > 0) {
				const Tensor shuffledX = reshuffle(x.get_core(_setId), {1, 0, 2});
				
				#pragma omp parallel for firstprivate(dyadComp, tmp)
				for(size_t jIdx = 0; jIdx < sets[_setId].size(); ++jIdx) {
					const size_t j = sets[_setId][jIdx];
					
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
				Tensor shuffledX = x.get_core(_setId);
				shuffledX.reinterpret_dimensions({shuffledX.dimensions[1], shuffledX.dimensions[2]});
				
				#pragma omp parallel for  firstprivate(dyadComp, tmp)
				for(size_t jIdx = 0; jIdx < sets[_setId].size(); ++jIdx) {
					const size_t j = sets[_setId][jIdx];
					contract(dyadComp, shuffledX, rightStack[_corePosition+1][j], 1);
					contract(dyadComp, dyadComp - solutions[j], rightStack[_corePosition+1][j], 0);
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
// 				#pragma omp parallel for firstprivate(tmp) reduction(+:norm)
				for(size_t jIdx = 0; jIdx < sets[_setId].size(); ++jIdx) {
					const size_t j = sets[_setId][jIdx];
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
				#pragma omp parallel for firstprivate(tmp, rightPart) reduction(+:norm)
				for(size_t jIdx = 0; jIdx < sets[_setId].size(); ++jIdx) {
					const size_t j = sets[_setId][jIdx];
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
		
		
		
		std::tuple<double, double, std::vector<double>> calc_residuals(const size_t _corePosition) const {
			REQUIRE(_corePosition == 0, "Invalid corePosition");

			double optResidual = 0.0;
			double testResidual = 0.0;
			std::vector<double> setResiduals(sets.size(), 0.0);
			
			const auto avgCore = x.get_average_core();
			Tensor tmp;
			#pragma omp parallel for firstprivate(tmp) reduction(+:optResidual, testResidual)
			for(size_t j = 0; j < N; ++j) {
				contract(tmp, avgCore, rightStack[1][j], 1);
				tmp.reinterpret_dimensions({x.dimensions[0]});
				tmp -= solutions[j];
				const double resSqr = misc::sqr(frob_norm(tmp));
				
				if(misc::contains(controlSet, j)){
					testResidual += resSqr;
				} else {
					optResidual += resSqr;
					
					for(size_t k = 0; k < sets.size(); ++k) {
						if(misc::contains(sets[k], j)) {
							#pragma omp critical
							{
								setResiduals[k] += resSqr;
							}
						}
					}
				}
			}
			
			optResidual = std::sqrt(optResidual)/optNorm;
			testResidual = std::sqrt(testResidual)/testNorm;
			for(size_t k = 0; k < sets.size(); ++k) {
				setResiduals[k] = std::sqrt(setResiduals[k])/setNorms[k];
			}
			
			
			return std::make_tuple(optResidual, testResidual, setResiduals);
		}
		
		void update_core(const size_t _corePosition) {
			const Index left, right, ext, p;
			
			for(size_t setId = 0; setId < P; ++setId) {
				const auto delta = calculate_delta(_corePosition, setId);
				const auto normAProjGrad = calculate_norm_A_projGrad(delta, _corePosition, setId);
				const value_t PyR = misc::sqr(frob_norm(delta));
				
				// Actual update
				x.component(_corePosition)(left, ext, p, right) = x.component(_corePosition)(left, ext, p, right)-((PyR/misc::sqr(normAProjGrad))*delta)(left, ext, right)*Tensor::dirac({P}, setId)(p);
			}
		}
		
		void finish() {
			for(size_t i = 0; i < bestX.degree(); i++) {
				if(i == bestX.corePosition) {
					outX.set_component(i, bestX.get_average_core());
				} else {
					outX.set_component(i, bestX.get_component(i));
				}
			}
			
			LOG(ADF, "Residual decrease from " << std::scientific << residuals[10] << " to " << std::scientific << residuals.back() << " in " << residuals.size()-10 << " iterations.");
		}
		
		
		void solve() {
			size_t nonImprovementCounter;
			
			// Build inital right stack
			REQUIRE(x.corePosition == 0, "Expecting core position to be 0.");
			for(size_t corePosition = d-1; corePosition > 0; --corePosition) {
				calc_right_stack(corePosition);
			}
			
			for(size_t iteration = 0; maxIterations == 0 || iteration < maxIterations; ++iteration) {
				double optResidual, testResidual;
				std::vector<double> setResiduals;
				std::tie(optResidual, testResidual, setResiduals) = calc_residuals(0);
				residuals.push_back(optResidual);
				
				if(testResidual < bestTestResidual) {
					bestX = x;
					bestTestResidual = testResidual;
					nonImprovementCounter = 0;
				} else {
					nonImprovementCounter++;
				}
				
						
				LOG(ADFx, "Residual " << std::scientific << residuals.back() << " " << setResiduals << " . Controlset: " << testResidual << ". Ranks: " << x.ranks() << ". DOFs: " << x.dofs());
				
				if(residuals.back() < targetResidual) {
					finish();
					return;
				}
				
				if(residuals.back()/residuals[residuals.size()-10] > convergenceFactor) {
					if(nonImprovementCounter > 100 || rankEps == minRankEps) {
						finish();
						return; // We are done!
					}
					LOG(ADFx, "Reduce rankEps to " << epsDecay*rankEps);
					rankEps = std::max(minRankEps, epsDecay*rankEps);
				}
					
				// Forward sweep
				for(size_t corePosition = 0; corePosition+1 < d; ++corePosition) {
					update_core(corePosition);
					
					x.move_core(corePosition+1, rankEps, std::min(maxRank, x.rank(corePosition)+1));
					calc_left_stack(corePosition);
				}
				
				update_core(d-1);
				
				// Backward sweep
				for(size_t corePosition = d-1; corePosition > 0; --corePosition) {
					update_core(corePosition);
					
					x.move_core(corePosition-1, rankEps, std::min(maxRank, x.rank(corePosition-1)+1));
					calc_right_stack(corePosition);
				}
				
				update_core(0);
			}
			
			finish();
		}
	};
	
}
	
	
	TTTensor uq_ra_adf(const UQMeasurementSet& _measurments, const PolynomBasis _basisType, const std::vector<size_t>& _dimensions, const double _initalRankEps, const double _targetEps, const size_t _maxItr) {
		REQUIRE(_measurments.randomVectors.size() == _measurments.solutions.size(), "Invalid measurments");
		REQUIRE(_measurments.initialRandomVectors.size() == _measurments.initialSolutions.size(), "Invalid initial measurments");
		REQUIRE(_dimensions.front() == _measurments.solutions.front().size, "Inconsitent spacial dimension");
		
		LOG(RankAdaptiveUQ, "Calculating Average as start.");

		TTTensor x(_dimensions);
			
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
		
// 		x = initial_guess(_measurments, x);
		
		impl_uqRaAdf::InternalSolver solver(x, _measurments.randomVectors, _measurments.solutions, _basisType, _maxItr, _targetEps, _initalRankEps);
		solver.solve();
		return x;
	}
	
	
	TTTensor uq_ra_adf(const UQMeasurementSet& _measurments, const PolynomBasis _basisType, const TTTensor& _initalGuess, const double _initalRankEps, const double _targetEps, const size_t _maxItr) {
		REQUIRE(_measurments.randomVectors.size() == _measurments.solutions.size(), "Invalid measurments");
		REQUIRE(_measurments.initialRandomVectors.size() == _measurments.initialSolutions.size(), "Invalid initial measurments");
		REQUIRE(_initalGuess.dimensions.front() == _measurments.solutions.front().size, "Inconsitent spacial dimension");
		
		TTTensor x = _initalGuess;
		
		impl_uqRaAdf::InternalSolver solver(x, _measurments.randomVectors, _measurments.solutions, _basisType, _maxItr, _targetEps, _initalRankEps);
		solver.solve();
		return x;
	}

	
}} // namespace  uq | xerus
