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

#include <xerus/algorithms/asd.h>
#include <xerus/blockTT.h>
#include <xerus/misc/basicArraySupport.h>
#include <xerus/misc/internal.h>

#include <boost/circular_buffer.hpp>

#ifdef _OPENMP
	#include <omp.h>
#endif

namespace xerus { namespace impl_TrASD {

	const size_t tracking = 20;
	
	template <size_t P>
	class InternalSolver {
	private:
		///@brief Reference to the external solution (external ownership).
		TTTensor& outX;
		
		internal::BlockTT x;
		
		///@brief Degree of the solution.
		const size_t degree;
		
		///@brief Reference to the measurment set (external ownership)
		const RankOneMeasurementSet& measurments;
		
		///@brief Number of measurments (i.e. measurments.size())
		const size_t numMeasurments;
		
		
		///@brief Maximal allowed number of iterations (one iteration = one sweep)
		const size_t maxIterations; 
		
		///@brief The target residual norm at which the algorithm shall stop.
		const double targetResidualNorm;
		
		///@brief Minimal relative decrease of the residual norm ( (oldRes-newRes)/oldRes ) until either the ranks are increased (if allowed) or the algorithm stops.
		const double minimalResidualNormDecrease;
		
		const double minRankEps = 1e-8;
		
		const double epsDecay = 0.8;
		
		///@brief Maximally allowed ranks.
		const std::vector<size_t> maxRanks;
		
		///@brief Current rankEps
		double rankEps = 0.001;
		
		
		///@brief: Reference to the performanceData object (external ownership)
		PerformanceData& perfData;
		
		
		const double controlSetFraction = 0.1;
		
		///@brief Vector of measurment IDs for each set (0 to P-1). The set P is the control set.
		std::vector<std::vector<size_t>> sets;
		
		///@brief L2 Norms of the measured values for each set (0-P)
		std::vector<double> setNorms;
		
		///@brief L2 Norm of the optimizing sets (0-P-1).
		double optNorm;
		
		///@brief Best control residual found so far.
		double bestTestResidual;
		
		///@brief Solution, realizing the current best control residual
		internal::BlockTT bestX;
		
		
		boost::circular_buffer<std::vector<size_t>> prevRanks;
		
		boost::circular_buffer<double> residuals;
		
		///@brief Stack of pre calculated components from corePosition 0 to d-1
		std::vector<std::vector<Tensor>> leftStack;
		
		///@brief Stack of pre calculated components from corePosition 0 to d-1
		std::vector<std::vector<Tensor>> rightStack;
		
	public:
		InternalSolver(	TTTensor& _x,
						const RankOneMeasurementSet& _measurments,
						const size_t _maxIteration,
						const double _targetResidualNorm,
						const double _minimalResidualNormDecrease,
						const std::vector<size_t>& _maxRanks,
						PerformanceData& _perfData ) :
			outX(_x),
			x(_x, 0, P),
			degree(_x.degree()),
			
			measurments(_measurments),
			numMeasurments(_measurments.size()),
			
			maxIterations(_maxIteration),
			targetResidualNorm(_targetResidualNorm),
			minimalResidualNormDecrease(_minimalResidualNormDecrease),
			maxRanks(TTTensor::reduce_to_maximal_ranks(_maxRanks, _x.dimensions)),
			perfData(_perfData),
			
			sets(P+1),
			setNorms(P+1),
			
			bestTestResidual(std::numeric_limits<double>::max()),
			prevRanks(tracking, outX.ranks()),
			residuals(tracking, std::numeric_limits<double>::max()),
			
			leftStack(degree, std::vector<Tensor>(numMeasurments)),
			rightStack(degree, std::vector<Tensor>(numMeasurments))
			
			{
				_x.require_correct_format();
				XERUS_REQUIRE(numMeasurments > 0, "Need at very least one measurment.");
				XERUS_REQUIRE(measurments.degree() == degree, "Measurment degree must coincide with x degree.");
				
				//Create P sets used for optimization + one test set
				std::uniform_real_distribution<double> stochDist(0.0, 1.0);
				std::uniform_int_distribution<size_t> setDist(0, P-1);
				
				for(size_t i = 0; i < numMeasurments; ++i ) {
					if(stochDist(misc::randomEngine) > controlSetFraction) {
						sets[setDist(misc::randomEngine)].push_back( i );
					} else {
						sets[P].push_back(i);
					}
				}
				
				// Calculate the measurment norms
				optNorm = 0.0;
				for(size_t setId = 0; setId < P+1; ++setId ) {
					setNorms[setId] = 0.0;
					for(const auto i : sets[setId]) {
						setNorms[setId] += misc::sqr(measurments.measuredValues[i]);
					}
					if(setId < P) { optNorm += setNorms[setId]; }
					setNorms[setId] = std::sqrt(setNorms[setId]);
				}
				optNorm = std::sqrt(optNorm);
			}
			
	private:
		
		///@brief For each measurment sets the forwardStack at the given _corePosition to the contraction between the forwardStack at the previous corePosition (i.e. -1)
		/// and the given component contracted with the component of the measurment operator. For _corePosition == corePosition and _currentComponent == x.components(corePosition)
		/// this really updates the stack, otherwise it uses the stack as scratch space.
		void update_left_stack(const size_t _position, const Tensor& _currentComponent) {
			REQUIRE(_position < degree, "Invalid position");
			Tensor measCmp;
			
			if(_position > 0 && _position+1 < degree) {
				const Tensor shuffledX = reshuffle(_currentComponent, {1, 0, 2});
				for(size_t i = 0; i < numMeasurments; ++i ) {
					contract(measCmp, measurments.positions[i][_position], shuffledX, 1);
					contract(leftStack[_position][i], leftStack[_position-1][i], measCmp, 1);
				}
			} else if(_position == 0) {
				Tensor shuffledX = _currentComponent;
				shuffledX.reinterpret_dimensions({shuffledX.dimensions[1], shuffledX.dimensions[2]}); // Remove dangling 1-mode
				for(size_t i = 0; i < numMeasurments; ++i ) {
					contract(leftStack[_position][i], measurments.positions[i][_position], shuffledX, 1);
				}
			} else { // _position == d-1
				Tensor shuffledX = _currentComponent;
				shuffledX.reinterpret_dimensions({shuffledX.dimensions[0], shuffledX.dimensions[1]}); // Remove dangling 1-mode
				for(size_t i = 0; i < numMeasurments; ++i ) {
					contract(measCmp, shuffledX, measurments.positions[i][_position], 1);
					contract(leftStack[_position][i], leftStack[_position-1][i], measCmp, 1);
				}
			}
		}
		
		///@brief For each measurment sets the backwardStack at the given _corePosition to the contraction between the backwardStack at the previous corePosition (i.e. +1)
		/// and the given component contracted with the component of the measurment operator. For _corePosition == corePosition and _currentComponent == x.components(corePosition)
		/// this really updates the stack, otherwise it uses the stack as scratch space.
		void update_right_stack(const size_t _position, const Tensor& _currentComponent) {
			REQUIRE(_position < degree, "Invalid position");
			Tensor measCmp;
			
			
			if(_position > 0 && _position+1 < degree) {
				const Tensor shuffledX = reshuffle(_currentComponent, {1, 0, 2});
				for(size_t i = 0; i < numMeasurments; ++i ) {
					contract(measCmp, measurments.positions[i][_position], shuffledX, 1);
					contract(rightStack[_position][i], measCmp, rightStack[_position+1][i], 1);
				}
			} else if(_position == 0) {
				Tensor shuffledX = _currentComponent;
				shuffledX.reinterpret_dimensions({shuffledX.dimensions[1], shuffledX.dimensions[2]}); // Remove dangling 1-mode
				for(size_t i = 0; i < numMeasurments; ++i ) {
					contract(measCmp, measurments.positions[i][_position], shuffledX, 1);
					contract(rightStack[_position][i], measCmp, rightStack[_position+1][i], 1);
				}
			} else { // _position == d-1
				Tensor shuffledX = _currentComponent;
				shuffledX.reinterpret_dimensions({shuffledX.dimensions[0], shuffledX.dimensions[1]}); // Remove dangling 1-mode
				for(size_t i = 0; i < numMeasurments; ++i ) {
					contract(rightStack[_position][i], shuffledX, measurments.positions[i][_position], 1);
				}
			}
		}
		
		
		///@brief Calculates the current residual, i.e. Ax-b, for a given set.
		std::vector<value_t> calculate_residual( const size_t _corePosition, const size_t _setId) {
			std::vector<value_t> residual(sets[_setId].size());
			
			Tensor leftCore, leftCorePos, leftCorePosRight;
			if( _corePosition > 0 && _corePosition+1 < degree) {
				for(size_t idx = 0; idx < sets[_setId].size(); ++idx) {
					const size_t i = sets[_setId][idx];
					contract(leftCore, leftStack[_corePosition-1][i], x.get_core(_setId), 1);
					contract(leftCorePos, measurments.positions[i][_corePosition], leftCore, 1);
					contract(leftCorePosRight, leftCorePos, rightStack[_corePosition+1][i], 1);
					residual[idx] = leftCorePosRight[0] - measurments.measuredValues[i];
					REQUIRE(leftCorePosRight.size == 1, "IE");
				}
			} else if( _corePosition == 0) {
				leftCore = x.get_core(_setId);
				leftCore.reinterpret_dimensions({leftCore.dimensions[1], leftCore.dimensions[2]}); // Remove dangling 1-mode
				for(size_t idx = 0; idx < sets[_setId].size(); ++idx) {
					const size_t i = sets[_setId][idx];
					contract(leftCorePos, measurments.positions[i][_corePosition], leftCore, 1);
					contract(leftCorePosRight, leftCorePos, rightStack[_corePosition+1][i], 1);
					residual[idx] = leftCorePosRight[0] - measurments.measuredValues[i];
					REQUIRE(leftCorePosRight.size == 1, "IE");
				}
			} else { // _position == d-1
				Tensor shuffledX = x.get_core(_setId);
				shuffledX.reinterpret_dimensions({shuffledX.dimensions[0], shuffledX.dimensions[1]}); // Remove dangling 1-mode
				for(size_t idx = 0; idx < sets[_setId].size(); ++idx) {
					const size_t i = sets[_setId][idx];
					contract(leftCore, leftStack[_corePosition-1][i], shuffledX, 1);
					contract(leftCorePos, measurments.positions[i][_corePosition], leftCore, 1);
					residual[idx] = leftCorePos[0] - measurments.measuredValues[i];
					REQUIRE(leftCorePos.size == 1, "IE");
				}
			}
			
			return residual;
		}
		
		
		///@brief: Calculates the component at _corePosition of the projected gradient from the residual, i.e. E(A^T(b-Ax)).
		Tensor calculate_delta(const size_t _corePosition, const size_t _setId) {
			const size_t localLeftRank = _corePosition == 0 ? 1 : x.rank(_corePosition-1);
			const size_t localRightRank = _corePosition+1 == degree ? 1 : x.rank(_corePosition);
			const size_t dyadDim = localLeftRank*localRightRank;
			
			const auto residual = calculate_residual(_corePosition, _setId);
			
			Tensor delta({x.dimensions[_corePosition], localLeftRank, localRightRank}, Tensor::Representation::Dense);
			delta.ensure_own_data_and_apply_factor();
			std::unique_ptr<value_t[]> dyadicComponent(new value_t[dyadDim]);
			
			for(size_t idx = 0; idx < sets[_setId].size(); ++idx) {
				const size_t i = sets[_setId][idx];
				if(_corePosition > 0 && _corePosition+1 < degree) {
					for(size_t k = 0; k < localLeftRank; ++k) {
						for(size_t j = 0; j < localRightRank; ++j) {
							dyadicComponent[k*localRightRank+j] = leftStack[_corePosition-1][i][k] * rightStack[_corePosition+1][i][j];
						}
					}
				} else if(_corePosition == 0) {
					for(size_t j = 0; j < localRightRank; ++j) {
						dyadicComponent[j] = rightStack[_corePosition+1][i][j];
					}
				} else { // _corePosition == d-1
					for(size_t k = 0; k < localLeftRank; ++k) {
						dyadicComponent[k] = leftStack[_corePosition-1][i][k];
					}
				}
				
				for(size_t n = 0; n < measurments.positions[i][_corePosition].size; ++n) {
					misc::add_scaled( delta.get_unsanitized_dense_data() + n*dyadDim,
						measurments.positions[i][_corePosition][n]*residual[idx],
						dyadicComponent.get(),
						dyadDim
					);
				}
			}
			
			delta = reshuffle( delta, {1,0,2});
			
			return delta;
		}
		
		/**
		* @brief: Calculates ||(A(E(A^T(b-Ax)))))|| = ||(A(E(A^T(residual)))))|| =  ||(A(E(gradient)))||.
		*/
		value_t calculate_norm_A_projGrad( const size_t _corePosition, const size_t _setId, const Tensor& _delta) {
			value_t normSqrAProjGrad = 0.0;
			
			Tensor leftCore, leftCorePos, leftCorePosRight;
			if( _corePosition > 0 && _corePosition+1 < degree) {
				for(size_t idx = 0; idx < sets[_setId].size(); ++idx) {
					const size_t i = sets[_setId][idx];
					contract(leftCore, leftStack[_corePosition-1][i], _delta, 1);
					contract(leftCorePos, measurments.positions[i][_corePosition], leftCore, 1);
					contract(leftCorePosRight, leftCorePos, rightStack[_corePosition+1][i], 1);
					normSqrAProjGrad += misc::sqr(leftCorePosRight[0]);
					REQUIRE(leftCorePosRight.size == 1, "IE");
				}
			} else if( _corePosition == 0) {
				leftCore = _delta;
				leftCore.reinterpret_dimensions({leftCore.dimensions[1], leftCore.dimensions[2]}); // Remove dangling 1-mode
				for(size_t idx = 0; idx < sets[_setId].size(); ++idx) {
					const size_t i = sets[_setId][idx];
					contract(leftCorePos, measurments.positions[i][_corePosition], leftCore, 1);
					contract(leftCorePosRight, leftCorePos, rightStack[_corePosition+1][i], 1);
					normSqrAProjGrad += misc::sqr(leftCorePosRight[0]);
					REQUIRE(leftCorePosRight.size == 1, "IE");
				}
			} else { // _position == d-1
				Tensor shuffledX = _delta;
				shuffledX.reinterpret_dimensions({shuffledX.dimensions[0], shuffledX.dimensions[1]}); // Remove dangling 1-mode
				for(size_t idx = 0; idx < sets[_setId].size(); ++idx) {
					const size_t i = sets[_setId][idx];
					contract(leftCore, leftStack[_corePosition-1][i], shuffledX, 1);
					contract(leftCorePos, measurments.positions[i][_corePosition], leftCore, 1);
					normSqrAProjGrad += misc::sqr(leftCorePos[0]);
					REQUIRE(leftCorePos.size == 1, "IE");
				}
			}
			
			return std::sqrt(normSqrAProjGrad);
		}
		
		
		void update_core(const size_t _corePosition) {
			const Index left, right, ext, p;
			
			for(size_t setId = 0; setId < P; ++setId) {
				const auto delta = calculate_delta(_corePosition, setId);
				const auto normAProjGrad = calculate_norm_A_projGrad(_corePosition, setId, delta);
				const value_t PyR = misc::sqr(frob_norm(delta));
				
				// Actual update
				x.component(_corePosition)(left, ext, p, right) = x.component(_corePosition)(left, ext, p, right)-((PyR/misc::sqr(normAProjGrad))*delta)(left, ext, right)*Tensor::dirac({P}, setId)(p);
			}
		}
		
		void finish(const size_t _iteration) {
			for(size_t i = 0; i < bestX.degree(); i++) {
				if(i == bestX.corePosition) {
					outX.set_component(i, bestX.get_average_core());
				} else {
					outX.set_component(i, bestX.get_component(i));
				}
			}
			
			LOG(ADF, "Residual decrease from " << std::scientific << 0.0 /* TODO */ << " to " << std::scientific << residuals.back() << " in " << _iteration << " iterations.");
		}
		
		std::pair<double, double> calc_residuals(const size_t _corePosition) const {
			REQUIRE(_corePosition == 0, "Invalid corePosition");
			
			auto avgCore = x.get_average_core();
			avgCore.reinterpret_dimensions({avgCore.dimensions[1], avgCore.dimensions[2]});
			Tensor tmp, tmp2;
			
			double optResidual = 0.0;			
			for(size_t p = 0; p < P; ++p ) {
				for(const auto i : sets[p]) {
					contract(tmp, avgCore, rightStack[_corePosition+1][i], 1);
					contract(tmp2, measurments.positions[i][_corePosition], tmp, 1);
					optResidual += misc::sqr(tmp2[0] - measurments.measuredValues[i]);
					REQUIRE(tmp2.size == 1, "IE");
				}
			}
			optResidual = std::sqrt(optResidual)/optNorm;
			
			double testResidual = 0.0;		
			for(const auto i : sets[P]) {
				contract(tmp, avgCore, rightStack[_corePosition+1][i], 1);
				contract(tmp2, measurments.positions[i][_corePosition], tmp, 1);
				testResidual += misc::sqr(tmp2[0] - measurments.measuredValues[i]);
				REQUIRE(tmp2.size == 1, "IE");
			}
			testResidual = std::sqrt(testResidual)/setNorms[P];
			
			return std::make_pair(optResidual, testResidual);
		}
		
	public:
		
		void solve() {
			size_t nonImprovementCounter = 0;
			
			// Build inital right stack
			REQUIRE(x.corePosition == 0, "Expecting core position to be 0.");
			for(size_t corePosition = degree-1; corePosition > 0; --corePosition) {
				update_right_stack(corePosition, x.get_component(corePosition));
			}
			
			for(size_t iteration = 0; maxIterations == 0 || iteration < maxIterations; ++iteration) {
				double optResidual, testResidual;
				std::tie(optResidual, testResidual) = calc_residuals(0);
				residuals.push_back(optResidual);
				prevRanks.push_back(x.ranks());
				
				if(testResidual < 0.9999*bestTestResidual) {
					bestX = x;
					bestTestResidual = testResidual;
					nonImprovementCounter = 0;
				} else {
					nonImprovementCounter++;
				}
				
						
				LOG(ADFx, "Residual " << std::scientific << residuals.back() << " " << /*setResiduals*/ -1 << ". NonImpCnt: " << nonImprovementCounter << ", Controlset: " << testResidual << ". Ranks: " << x.ranks() << ". DOFs: " << x.dofs() << ". Norm: " << frob_norm(x.get_average_core()));
				
				if(residuals.back() < targetResidualNorm || nonImprovementCounter >= 100) {
					finish(iteration);
					LOG(adoff, "Target or MaxIter");
					return;
				}
				
				if(residuals.back() > minimalResidualNormDecrease*residuals[0]) {
					bool maxRankReached = false;
					bool rankMaxed = false;
					for(size_t k = 0; k < x.degree()-1; ++k ) {
						maxRankReached = maxRankReached || (x.rank( k ) == maxRanks[k]);
						rankMaxed = rankMaxed || (x.rank( k ) == prevRanks[0][k]+1);
					}
					
					if(misc::hard_equal(rankEps, minRankEps) || maxRankReached) {
						LOG(adoff, "MinRankEps or maxRank.");
						finish(iteration);
						return; // We are done!
					}
					
					if(!rankMaxed) {
						LOG(ADFx, "Reduce rankEps to " << std::max(minRankEps, epsDecay*rankEps));
						rankEps = std::max(minRankEps, epsDecay*rankEps);
					}
				}
				
				for(size_t i = 0; i < x.degree(); i++) {
					if(i == x.corePosition) {
						outX.set_component(i, x.get_average_core());
					} else {
						outX.set_component(i, x.get_component(i));
					}
				}
				
				perfData.add(iteration, testResidual, outX, 0);
					
				// Forward sweep
				for(size_t corePosition = 0; corePosition+1 < degree; ++corePosition) {
					update_core(corePosition);
					
					
					x.move_core_right(rankEps, std::min(maxRanks[corePosition], prevRanks[1][corePosition]+1));
					update_left_stack(corePosition, x.get_component(corePosition));
				}
				
				update_core(degree-1);
				
				
				// Backward sweep
				for(size_t corePosition = degree-1; corePosition > 0; --corePosition) {
// 					update_core(corePosition);
					
					x.move_core_left(rankEps, std::min(maxRanks[corePosition-1], prevRanks[1][corePosition-1]+1));
					update_right_stack(corePosition, x.get_component(corePosition));
				}
				
// 				update_core(0);
			}
			
			finish(maxIterations);
		}	
	};
	
} // namespace impl_TrASD


	void ASDVariant::operator()(TTTensor& _x, const RankOneMeasurementSet& _measurments, PerformanceData& _perfData) const {
		impl_TrASD::InternalSolver<2> solver(_x, _measurments, maxIterations, targetRelativeResidual, minimalResidualNormDecrease, _x.ranks(), _perfData);
		solver.solve();
	}
	

	void ASDVariant::operator()(TTTensor& _x, const RankOneMeasurementSet& _measurments, const std::vector<size_t>& _maxRanks, PerformanceData& _perfData) const {
		impl_TrASD::InternalSolver<2> solver(_x, _measurments, maxIterations, targetRelativeResidual, minimalResidualNormDecrease, _maxRanks, _perfData);
		solver.solve();
	}
	
	// Explicit instantiation of the two template parameters that will be implemented in the xerus library
// 	template class ASDVariant::InternalSolver;
	
	const ASDVariant TRASD(0, 1e-8, 0.999);
} // namespace xerus
