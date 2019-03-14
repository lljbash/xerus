// Xerus - A General Purpose Tensor Library
// Copyright (C) 2014-2019 Benjamin Huber and Sebastian Wolf. 
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
* @brief Header file for the OptimizationAlgorithm class.
*/

#pragma once

#include "../basic.h"
#include "../forwardDeclarations.h"

#include <boost/circular_buffer.hpp>



namespace xerus {

	/**
	* @brief Base class for (in future) all xerus optimization algorithms, allowing a uniform set of settings.
	*/
	class OptimizationAlgorithm {
	public:
		///@brief Minimal number of iterations.
		size_t minIterations;
		
		///@brief Maximal allowed number of iterations. Zero for infinite.
		size_t maxIterations;
		
		///@brief The target residual norm at which the algorithm shall stop.
		double targetRelativeResidual;
		
		///@brief Minimal decrease of the residual norm ( newRes/oldRes ) until either the ranks are increased (if allowed) or the algorithm stops.
		double minimalResidualDecrease;
		
		///@brief Number of iterations used to check for stopping criteria (e.g. residual[iterations] <= residual[iteration-tracking]*pow(minimalResidualDecrease, tracking) )
		size_t tracking = 10;
		
		
	protected:
		OptimizationAlgorithm(const size_t _minIterations, const size_t _maxIterations, const double _targetRelativeResidual, const double _minimalResidualDecrease);
	};
	
	
	namespace internal {
		class OptimizationSolver {
		protected:
			///@brief Minimal number of iterations.
			const size_t minIterations;
			
			///@brief Maximal allowed number of iterations. Zero for infinite.
			const size_t maxIterations;
			
			///@brief The target residual norm at which the algorithm shall stop.
			const double targetRelativeResidual;
			
			///@brief Minimal decrease of the residual norm ( newRes/oldRes ) until either the ranks are increased (if allowed) or the algorithm stops.
			const double minimalResidualDecrease;
			
			///@brief Number of iterations used to check for stopping criteria (e.g. residual[iterations] <= residual[iteration-tracking]*pow(minimalResidualDecrease, tracking) )
			const size_t tracking;
			
			///@brief Defined as pow(minimalResidualDecrease, tracking).
			const double convergenceFactor;
			
		private:
			///@brief The current iteration.
			size_t iteration = 0;
			
			///@brief The last (tracking) residuals.
			boost::circular_buffer<double> lastResiduals;
			
		protected:
			///@brief: Reference to the performanceData object (external ownership)
			PerformanceData& perfData;
			
		
			OptimizationSolver(const OptimizationAlgorithm& _optiAlgorithm, PerformanceData& _perfData);
			
			
			///@brief Increased iteration by one and adds the residual to the circular buffer.
			void make_step(const double _residual);
			
			size_t current_iteration() const;
			
			double current_residual() const;
			
			///@brief True if either the maxIterations are reached or the targetRelativeResidual is reached.
			bool reached_stopping_criteria() const;
			
			///@brief True if either the minInterations are reached and convegence is reached (i.e. residual[iterations] <= residual[iteration-tracking]*pow(minimalResidualDecrease, tracking) ).
			bool reached_convergence_criteria() const;
			
			///@brief Resets the convergence buffer with max doubles. In particular at least tracking iterations are then nessecary the reach convergence.
			void reset_convergence_buffer();
		};
	
	} // End namespace internal
} // End namespace xerus
