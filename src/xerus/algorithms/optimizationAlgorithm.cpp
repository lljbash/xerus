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
* @brief Implementation of the OptimizationAlgorithm class.
*/

#include <xerus/algorithms/optimizationAlgorithm.h>

#include <xerus/misc/math.h>

#include <xerus/performanceData.h>

namespace xerus {
	OptimizationAlgorithm::OptimizationAlgorithm(const size_t _minIterations, const size_t _maxIterations, const double _targetRelativeResidual, const double _minimalResidualDecrease) : 
	minIterations(_minIterations), 
	maxIterations(_maxIterations), 
	targetRelativeResidual(_targetRelativeResidual), 
	minimalResidualDecrease(_minimalResidualDecrease)
	{}
	
	
	internal::OptimizationSolver::OptimizationSolver(const OptimizationAlgorithm& _optiAlgorithm, PerformanceData& _perfData) : 
	minIterations(_optiAlgorithm.minIterations), 
	maxIterations(_optiAlgorithm.maxIterations), 
	targetRelativeResidual(_optiAlgorithm.targetRelativeResidual), 
	minimalResidualDecrease(_optiAlgorithm.minimalResidualDecrease),
	tracking(_optiAlgorithm.tracking),
	convergenceFactor(misc::pow(minimalResidualDecrease, tracking)),
	lastResiduals(tracking, std::numeric_limits<double>::max()),
	perfData(_perfData)
	{ }
	
	
	void internal::OptimizationSolver::make_step(const double _residual) {
		iteration++;
		lastResiduals.push_back(_residual);
	}
	
	
	size_t internal::OptimizationSolver::current_iteration() const {
		return iteration;
	}
	
	
	double internal::OptimizationSolver::current_residual() const {
		return lastResiduals.back();
	}
	
	
	bool internal::OptimizationSolver::reached_stopping_criteria() const {
		return (maxIterations > 0 && iteration >= maxIterations) || (iteration >= minIterations && lastResiduals.back() <= targetRelativeResidual);
	}
	
	
	bool internal::OptimizationSolver::reached_convergence_criteria() const {
		return iteration >= minIterations && lastResiduals.back() > convergenceFactor*lastResiduals.front();
	}
	
	
	void internal::OptimizationSolver::reset_convergence_buffer() {
		lastResiduals = boost::circular_buffer<double>(tracking, std::numeric_limits<double>::max());
	}
	
} // namespace xerus
