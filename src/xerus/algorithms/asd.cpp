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

	
	template <size_t P>
	class InternalSolver {
	private:
		///@brief Reference to the external solution (external ownership).
		TTTensor& outX;
		
		///@brief The internal blockTT representation of the current solution.
		internal::BlockTT x;
		
		///@brief Degree of the solution.
		const size_t degree;
		
		///@brief Reference to the measurment set (external ownership)
		const RankOneMeasurementSet& measurments;
		
		///@brief Number of measurments (i.e. measurments.size())
		const size_t numMeasurments;
		
		///@brief Minimal number of iterations (one iteration = one sweep)
		const size_t minIterations;
		
		///@brief Maximal allowed number of iterations (one iteration = one sweep)
		const size_t maxIterations;
		
		///@brief The target residual norm at which the algorithm shall stop.
		const double targetRelativeResidual;
		
		///@brief Minimal relative decrease of the residual norm ( newRes/oldRes ) until either the ranks are increased (if allowed) or the algorithm stops.
		const double minimalResidualNormDecrease;
		
		const size_t tracking;
		
		///@brief Maximally allowed ranks.
		const std::vector<size_t> maxRanks;
		
		const double minRankEps;
		
		const double epsDecay;
		
		const double controlSetFraction;
		
		
		
		///@brief: Reference to the performanceData object (external ownership)
		PerformanceData& perfData;
		
		
		///@brief Vector of measurment IDs for each set (0 to P-1). The set P is the control set.
		std::vector<std::vector<size_t>> sets;
		
		///@brief L2 Norms of the measured values for each set (0-P)
		std::vector<double> setNorms;
		
		///@brief L2 Norm of the optimizing sets (0-P-1).
		double optNorm;
		
		
		///@brief Current rankEps
		double rankEps;
		
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
		
		void shuffle_sets() {
			//Reset all sets
			for(size_t setId = 0; setId < P; ++setId) {
				sets[setId].clear();
			}
			
			//Create the P sets used for optimization
			std::uniform_int_distribution<size_t> setDist(0, P-1);
			
			for(size_t i = 0; i < numMeasurments; ++i ) {
				if(!misc::contains(sets[P], i)) {
					sets[setDist(misc::randomEngine)].push_back( i );
				}
			}
			
			// Calculate the measurment norms
			for(size_t setId = 0; setId < P; ++setId) {
				setNorms[setId] = 0.0;
				for(const auto i : sets[setId]) {
					setNorms[setId] += misc::sqr(measurments.measuredValues[i]);
				}
				setNorms[setId] = std::sqrt(setNorms[setId]);
			}
		}
		
		InternalSolver(	TTTensor& _x,
						const RankOneMeasurementSet& _measurments,
						const ASDVariant& _optiSettings,
						const double _initalRankEps,
						const std::vector<size_t>& _maxRanks,
						PerformanceData& _perfData ) :
			outX(_x),
			x(_x, 0, P),
			degree(_x.degree()),
			
			measurments(_measurments),
			numMeasurments(_measurments.size()),
			
			minIterations(_optiSettings.minIterations),
			maxIterations(_optiSettings.maxIterations),
			targetRelativeResidual(_optiSettings.targetRelativeResidual),
			minimalResidualNormDecrease(_optiSettings.minimalResidualNormDecrease),
			tracking(_optiSettings.tracking),
			
			maxRanks(TTTensor::reduce_to_maximal_ranks(_maxRanks, _x.dimensions)),
			
			minRankEps(_optiSettings.minRankEps),
			epsDecay(_optiSettings.epsDecay),
			
			controlSetFraction(P == 1 ? 0.0 : _optiSettings.controlSetFraction),
			
			perfData(_perfData),
			
			
			sets(P+1),
			setNorms(P+1),
			
			rankEps(_initalRankEps),
			
			bestTestResidual(std::numeric_limits<double>::max()),
			prevRanks(tracking, outX.ranks()),
			residuals(tracking, std::numeric_limits<double>::max()),
			
			leftStack(degree, std::vector<Tensor>(numMeasurments)),
			rightStack(degree, std::vector<Tensor>(numMeasurments))
			
			{
				_x.require_correct_format();
				XERUS_REQUIRE(numMeasurments > 0, "Need at very least one measurment.");
				XERUS_REQUIRE(measurments.degree() == degree, "Measurment degree must coincide with x degree.");
				
				// Create test set
				std::uniform_real_distribution<double> stochDist(0.0, 1.0);
				
				optNorm = 0.0;
				setNorms[P] = 0.0;
				for(size_t i = 0; i < numMeasurments; ++i ) {
					if(stochDist(misc::randomEngine) < controlSetFraction) {
						sets[P].push_back(i);
						setNorms[P] += misc::sqr(measurments.measuredValues[i]);
					} else {
						optNorm += misc::sqr(measurments.measuredValues[i]);
					}
				}
				optNorm = std::sqrt(optNorm);
				setNorms[P] = std::sqrt(setNorms[P]);
				
				shuffle_sets();
			}
			
	private:
		
		///@brief For each measurment sets the forwardStack at the given _corePosition to the contraction between the forwardStack at the previous corePosition (i.e. -1)
		/// and the given component contracted with the component of the measurment operator. For _corePosition == corePosition and _currentComponent == x.components(corePosition)
		/// this really updates the stack, otherwise it uses the stack as scratch space.
		void update_left_stack(const size_t _position) {
			REQUIRE(_position+1 < degree, "Invalid position");
			Tensor measCmp;
			
			if(_position > 0 && _position+1 < degree) {
				const Tensor shuffledX = reshuffle(x.get_component(_position), {1, 0, 2});
				for(size_t i = 0; i < numMeasurments; ++i ) {
					contract(measCmp, measurments.positions[i][_position], shuffledX, 1);
					contract(leftStack[_position][i], leftStack[_position-1][i], measCmp, 1);
				}
			} else if(_position == 0) {
				Tensor shuffledX = x.get_component(_position);
				shuffledX.reinterpret_dimensions({shuffledX.dimensions[1], shuffledX.dimensions[2]}); // Remove dangling 1-mode
				for(size_t i = 0; i < numMeasurments; ++i ) {
					contract(leftStack[_position][i], measurments.positions[i][_position], shuffledX, 1);
				}
			}
		}
		
		///@brief For each measurment sets the backwardStack at the given _corePosition to the contraction between the backwardStack at the previous corePosition (i.e. +1)
		/// and the given component contracted with the component of the measurment operator. For _corePosition == corePosition and _currentComponent == x.components(corePosition)
		/// this really updates the stack, otherwise it uses the stack as scratch space.
		void update_right_stack(const size_t _position) {
			REQUIRE(_position > 0 && _position < degree, "Invalid position");
			Tensor measCmp;
			
			
			if(_position > 0 && _position+1 < degree) {
				const Tensor shuffledX = reshuffle(x.get_component(_position), {1, 0, 2});
				for(size_t i = 0; i < numMeasurments; ++i ) {
					contract(measCmp, measurments.positions[i][_position], shuffledX, 1);
					contract(rightStack[_position][i], measCmp, rightStack[_position+1][i], 1);
				}
			} else { // _position == d-1
				Tensor shuffledX = x.get_component(_position);
				shuffledX.reinterpret_dimensions({shuffledX.dimensions[0], shuffledX.dimensions[1]}); // Remove dangling 1-mode
				for(size_t i = 0; i < numMeasurments; ++i ) {
					contract(rightStack[_position][i], shuffledX, measurments.positions[i][_position], 1);
				}
			}
		}
		
		
		///@brief: Calculates the component at _corePosition of the projected gradient from the residual, i.e. E(A^T(b-Ax)).
		Tensor calculate_delta(const size_t _corePosition, const size_t _setId) {
			const size_t localLeftRank = _corePosition == 0 ? 1 : x.rank(_corePosition-1);
			const size_t localRightRank = _corePosition+1 == degree ? 1 : x.rank(_corePosition);
			const size_t dyadDim = localLeftRank*localRightRank;
			
			Tensor delta({x.dimensions[_corePosition], localLeftRank, localRightRank}, Tensor::Representation::Dense);
			std::unique_ptr<value_t[]> dyadicComponent(new value_t[dyadDim]);
			
			const Tensor core = x.get_core(_setId);
			Tensor leftCore, leftCorePos, leftCorePosRight;
			if( _corePosition > 0 && _corePosition+1 < degree) {
				for(size_t idx = 0; idx < sets[_setId].size(); ++idx) {
					const size_t i = sets[_setId][idx];
					contract(leftCore, leftStack[_corePosition-1][i], core, 1);
					contract(leftCorePos, measurments.positions[i][_corePosition], leftCore, 1);
					contract(leftCorePosRight, leftCorePos, rightStack[_corePosition+1][i], 1);
					const value_t residual = leftCorePosRight[0] - measurments.measuredValues[i];
					
					for(size_t k = 0; k < localLeftRank; ++k) {
						for(size_t j = 0; j < localRightRank; ++j) {
							dyadicComponent[k*localRightRank+j] = leftStack[_corePosition-1][i][k] * rightStack[_corePosition+1][i][j];
						}
					}
					
					for(size_t n = 0; n < measurments.positions[i][_corePosition].size; ++n) {
						misc::add_scaled( delta.get_unsanitized_dense_data() + n*dyadDim,
							measurments.positions[i][_corePosition][n]*residual,
							dyadicComponent.get(),
							dyadDim
						);
					}
				}
			} else if( _corePosition == 0) {
				leftCore = core;
				leftCore.reinterpret_dimensions({leftCore.dimensions[1], leftCore.dimensions[2]}); // Remove dangling 1-mode
				for(size_t idx = 0; idx < sets[_setId].size(); ++idx) {
					const size_t i = sets[_setId][idx];
					contract(leftCorePos, measurments.positions[i][_corePosition], leftCore, 1);
					contract(leftCorePosRight, leftCorePos, rightStack[_corePosition+1][i], 1);
					const value_t residual = leftCorePosRight[0] - measurments.measuredValues[i];
					
					for(size_t j = 0; j < localRightRank; ++j) {
						dyadicComponent[j] = rightStack[_corePosition+1][i][j];
					}
					
					for(size_t n = 0; n < measurments.positions[i][_corePosition].size; ++n) {
						misc::add_scaled( delta.get_unsanitized_dense_data() + n*dyadDim,
							measurments.positions[i][_corePosition][n]*residual,
							dyadicComponent.get(),
							dyadDim
						);
					}
				}
			} else { // _position == d-1
				for(size_t idx = 0; idx < sets[_setId].size(); ++idx) {
					const size_t i = sets[_setId][idx];
					contract(leftCore, leftStack[_corePosition-1][i], core, 1);
					contract(leftCorePos, measurments.positions[i][_corePosition], leftCore, 1);
					leftCorePosRight = leftCorePos;
					leftCorePosRight.reinterpret_dimensions({}); // Remove dangling 1-mode
					const value_t residual = leftCorePosRight[0] - measurments.measuredValues[i];
					
					for(size_t k = 0; k < localLeftRank; ++k) {
						dyadicComponent[k] = leftStack[_corePosition-1][i][k];
					}
					
					for(size_t n = 0; n < measurments.positions[i][_corePosition].size; ++n) {
						misc::add_scaled( delta.get_unsanitized_dense_data() + n*dyadDim,
							measurments.positions[i][_corePosition][n]*residual,
							dyadicComponent.get(),
							dyadDim
						);
					}
				}
			}
			
			return reshuffle( delta, {1,0,2} );
		}
		
		/**
		* @brief: Calculates ||(A(E(A^T(b-Ax)))))|| = ||(A(E(A^T(residual)))))|| =  ||(A(E(gradient)))||.
		*/
		value_t calculate_normSqr_A_projGrad( const size_t _corePosition, const size_t _setId, const Tensor& _delta) {
			Tensor leftCore, leftCorePos, leftCorePosRight;
			value_t normSqrAProjGrad = 0.0;
			
			if( _corePosition > 0 && _corePosition+1 < degree) {
				for(size_t idx = 0; idx < sets[_setId].size(); ++idx) {
					const size_t i = sets[_setId][idx];
					contract(leftCore, leftStack[_corePosition-1][i], _delta, 1);
					contract(leftCorePos, measurments.positions[i][_corePosition], leftCore, 1);
					contract(leftCorePosRight, leftCorePos, rightStack[_corePosition+1][i], 1);
					normSqrAProjGrad += misc::sqr(leftCorePosRight[0]);
				}
			} else if( _corePosition == 0) {
				leftCore = _delta;
				leftCore.reinterpret_dimensions({leftCore.dimensions[1], leftCore.dimensions[2]}); // Remove dangling 1-mode
				for(size_t idx = 0; idx < sets[_setId].size(); ++idx) {
					const size_t i = sets[_setId][idx];
					contract(leftCorePos, measurments.positions[i][_corePosition], leftCore, 1);
					contract(leftCorePosRight, leftCorePos, rightStack[_corePosition+1][i], 1);
					normSqrAProjGrad += misc::sqr(leftCorePosRight[0]);
				}
			} else { // _position == d-1
				for(size_t idx = 0; idx < sets[_setId].size(); ++idx) {
					const size_t i = sets[_setId][idx];
					contract(leftCore, leftStack[_corePosition-1][i], _delta, 1);
					contract(leftCorePos, measurments.positions[i][_corePosition], leftCore, 1);
					leftCorePosRight = leftCorePos;
					leftCorePosRight.reinterpret_dimensions({});// Remove dangling 1-mode
					normSqrAProjGrad += misc::sqr(leftCorePosRight[0]);
				}
			}
			
			return normSqrAProjGrad;
		}
		
		
		void update_core(const size_t _corePosition) {
			const Index left, right, ext, p;
			
			for(size_t setId = 0; setId < P; ++setId) {
				const auto delta = calculate_delta(_corePosition, setId);
				const value_t normSqrPyR = misc::sqr(frob_norm(delta));
				const auto normSqrAProjGrad = calculate_normSqr_A_projGrad(_corePosition, setId, delta);
				
				// Actual update
				x.component(_corePosition)(left, ext, p, right) = x.component(_corePosition)(left, ext, p, right)-((normSqrPyR/normSqrAProjGrad)*delta)(left, ext, right)*Tensor::dirac({P}, setId)(p);
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
		
		
		std::tuple<double, double, std::vector<double>> calc_residuals() const {
			REQUIRE(x.corePosition == 0, "Invalid corePosition");
			const size_t corePosition = 0;
			
			std::vector<double> setResiduals(P, 0.0);
			
			auto avgCore = x.get_average_core();
			avgCore.reinterpret_dimensions({avgCore.dimensions[1], avgCore.dimensions[2]});
			Tensor tmp, tmp2;
			
			double optResidual = 0.0;			
			for(size_t p = 0; p < P; ++p ) {
				auto setCore = x.get_core(p);
				setCore.reinterpret_dimensions({setCore.dimensions[1], setCore.dimensions[2]});
				
				for(const auto i : sets[p]) {
					contract(tmp, setCore, rightStack[corePosition+1][i], 1);
					contract(tmp2, measurments.positions[i][corePosition], tmp, 1);
					setResiduals[p] += misc::sqr(tmp2[0] - measurments.measuredValues[i]);
// 					const double test = tmp2[0];
					
					contract(tmp, avgCore, rightStack[corePosition+1][i], 1);
					contract(tmp2, measurments.positions[i][corePosition], tmp, 1);
					optResidual += misc::sqr(tmp2[0] - measurments.measuredValues[i]);
// 					LOG(bla, (test-tmp2[0])/test);
					REQUIRE(tmp2.size == 1, "IE");
				}
				setResiduals[p] = std::sqrt(setResiduals[p])/setNorms[p];
			}
			optResidual = std::sqrt(optResidual)/optNorm;
			
			double testResidual = 0.0;		
			for(const auto i : sets[P]) {
				contract(tmp, avgCore, rightStack[corePosition+1][i], 1);
				contract(tmp2, measurments.positions[i][corePosition], tmp, 1);
				testResidual += misc::sqr(tmp2[0] - measurments.measuredValues[i]);
				REQUIRE(tmp2.size == 1, "IE");
			}
			testResidual = std::sqrt(testResidual)/setNorms[P];
			
			return std::make_tuple(optResidual, testResidual, setResiduals);
		}
		
	public:
		
		void solve() {
			perfData.start();
			size_t nonImprovementCounter = 0;
			
			// Build inital right stack
			REQUIRE(x.corePosition == 0, "Expecting core position to be 0.");
			for(size_t corePosition = degree-1; corePosition > 0; --corePosition) {
				update_right_stack(corePosition);
			}
			
			for(size_t iteration = 0; maxIterations == 0 || iteration < maxIterations; ++iteration) {
				double optResidual, testResidual;
				std::vector<double> setResiduals;
				std::tie(optResidual, testResidual, setResiduals) = calc_residuals();
				
				residuals.push_back(optResidual);
				
// 				prevRanks.push_back(x.ranks());
				
				if(P == 1 || testResidual < 0.99*bestTestResidual) {
					bestX = x;
					bestTestResidual = testResidual;
					nonImprovementCounter = 0;
				} else {
					nonImprovementCounter++;
				}
				
				
// 				LOG(ADFx, "Residual " << std::scientific << residuals.back() << " " << /*setResiduals*/ -1 << ". NonImpCnt: " << nonImprovementCounter << ", Controlset: " << testResidual << ". Ranks: " << x.ranks() << ". DOFs: " << x.dofs() << ". Norm: " << frob_norm(x.get_average_core()));
				
// 				bool maxRankReached = true;
// 				for(size_t k = 0; k+1 < x.degree(); ++k ) {
// 					maxRankReached = maxRankReached && (x.rank(k) == maxRanks[k]);
// 				}
				
				if(P > 1 && nonImprovementCounter > 2) {
					rankEps = std::min(0.32, 2*rankEps);
					perfData << rankEps;
				}
				
				if(optResidual < targetRelativeResidual || nonImprovementCounter >= 25 || ( P == 1 && residuals.back() > std::pow(minimalResidualNormDecrease, tracking)*residuals[0])) {
					finish(iteration);
					return; // We are done!
				}

				
				perfData.add(iteration, std::vector<double>{optResidual, testResidual} | setResiduals, x.get_average_tt(), 0);
				
				if(P>1) { shuffle_sets(); }
					
				// Forward sweep
				for(size_t corePosition = 0; corePosition+1 < degree; ++corePosition) {
					update_core(corePosition);
					
					
					x.move_core_right(rankEps, /*std::min(*/maxRanks[corePosition]/*, prevRanks[0][corePosition]+1)*/);
					update_left_stack(corePosition);
				}
				
				update_core(degree-1);
				
				
				// Backward sweep
				for(size_t corePosition = degree-1; corePosition > 0; --corePosition) {
					update_core(corePosition);
					
					x.move_core_left(rankEps, /*std::min(*/maxRanks[corePosition-1]/*, prevRanks[0][corePosition-1]+1)*/);
					update_right_stack(corePosition);
				}
				
				update_core(0);
			}
			
			finish(maxIterations);
		}	
	};
	
} // namespace impl_TrASD


	void ASDVariant::operator()(TTTensor& _x, const RankOneMeasurementSet& _measurments, PerformanceData& _perfData) const {
		impl_TrASD::InternalSolver<1> solver(_x, _measurments, *this, initialRankEps, _x.ranks(), _perfData);
		solver.solve();
	}
	

	void ASDVariant::operator()(TTTensor& _x, const RankOneMeasurementSet& _measurments, const std::vector<size_t>& _maxRanks, PerformanceData& _perfData) const {
		impl_TrASD::InternalSolver<2> solver(_x, _measurments, *this, initialRankEps, _maxRanks, _perfData);
		solver.solve();
	}
	
	const ASDVariant TRASD(1000, 1e-10, 0.9995);
} // namespace xerus
