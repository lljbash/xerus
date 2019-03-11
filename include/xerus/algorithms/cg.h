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
 * @brief Header file for the CG algorithm.
 */

#pragma once

#include "../ttNetwork.h"
#include "../performanceData.h"
#include "retractions.h"

namespace xerus {

	/**
	* @brief Wrapper class for all geometric (ie. Riemannian) CG variants
	* @note only implemented for TTTensors at the moment.
	* @details By creating a new object of this class and modifying the member variables, the behaviour of the solver can be modified.
	* 
	* cf. Boumal, Absil. doi: 10.1016/j.laa.2015.02.027
	*  cf. Hager, Zhang. https://www.math.lsu.edu/~hozhang/papers/cgsurvey.pdf  for the beta rules
	*/
	class GeometricCGVariant {
	protected:
		double solve(const TTOperator *_Ap, TTTensor &_x, const TTTensor &_b, size_t _numSteps, value_t _convergenceEpsilon, PerformanceData &_perfData = NoPerfData) const;
	
	public:
		size_t numSteps; ///< maximum number of steps to perform. set to 0 for infinite
		value_t convergenceEpsilon; ///< default value for the change in the residual at which the algorithm assumes it is converged
		bool assumeSymmetricPositiveDefiniteOperator; ///< calculates the gradient as b-Ax instead of A^T(b-Ax)
		
		TTRetractionI retraction; ///< the retraction type I to project from point + tangent vector to a new point on the manifold
		TTVectorTransport vectorTransport; ///< the vector transport from old tangent space to new one
		
		// TODO preconditioner
		
		/// fully defining constructor. alternatively CGVariant can be created by copying a predefined variant and modifying it
		GeometricCGVariant(size_t _numSteps, value_t _convergenceEpsilon, bool _symPosOp,
						   TTRetractionI _retraction,
						   TTVectorTransport _vectorTransport
  						)
				: numSteps(_numSteps), convergenceEpsilon(_convergenceEpsilon), assumeSymmetricPositiveDefiniteOperator(_symPosOp), 
				  retraction(_retraction), vectorTransport(_vectorTransport)
		{ }
		
		/// definition using only the retraction. In the following an operator() including either convergenceEpsilon or numSteps must be called or the algorithm will never terminate
		GeometricCGVariant(TTRetractionI _retraction, TTVectorTransport _vectorTransport)
				: numSteps(0), convergenceEpsilon(0.0), assumeSymmetricPositiveDefiniteOperator(false), 
				  retraction(_retraction), vectorTransport(_vectorTransport)
		{ }
		
		/**
		* call to solve @f$ A\cdot x = b@f$ for @f$ x @f$ (in a least-squares sense)
		* @param _A operator to solve for
		* @param[in,out] _x in: initial guess, out: solution as found by the algorithm
		* @param _b right-hand side of the equation to be solved
		* @param _convergenceEpsilon minimum change in residual / energy under which the algorithm terminates
		* @param _perfData vector of performance data (residuals after every microiteration)
		* @returns the residual @f$|Ax-b|@f$ of the final @a _x
		*/
		double operator()(const TTOperator &_A, TTTensor &_x, const TTTensor &_b, value_t _convergenceEpsilon, PerformanceData &_perfData = NoPerfData) const {
			return solve(&_A, _x, _b, numSteps, _convergenceEpsilon, _perfData);
		}
		
		/**
		* call to solve @f$ A\cdot x = b@f$ for @f$ x @f$ (in a least-squares sense)
		* @param _A operator to solve for
		* @param[in,out] _x in: initial guess, out: solution as found by the algorithm
		* @param _b right-hand side of the equation to be solved
		* @param _numSteps maximum number of half-sweeps to perform
		* @param _perfData vector of performance data (residuals after every microiteration)
		* @returns the residual @f$|Ax-b|@f$ of the final @a _x
		*/
		double operator()(const TTOperator &_A, TTTensor &_x, const TTTensor &_b, size_t _numSteps, PerformanceData &_perfData = NoPerfData) const {
			return solve(&_A, _x, _b, _numSteps, convergenceEpsilon, _perfData);
		}
		
		/**
		* call to solve @f$ A\cdot x = b@f$ for @f$ x @f$ (in a least-squares sense)
		* @param _A operator to solve for
		* @param[in,out] _x in: initial guess, out: solution as found by the algorithm
		* @param _b right-hand side of the equation to be solved
		* @param _perfData vector of performance data (residuals after every microiteration)
		* @returns the residual @f$|Ax-b|@f$ of the final @a _x
		*/
		double operator()(const TTOperator &_A, TTTensor &_x, const TTTensor &_b, PerformanceData &_perfData = NoPerfData) const {
			return solve(&_A, _x, _b, numSteps, convergenceEpsilon, _perfData);
		}
		
		/**
		* call to minimze @f$ \|x - b\|^2 @f$ for @f$ x @f$
		* @param[in,out] _x in: initial guess, out: solution as found by the algorithm
		* @param _b right-hand side of the equation to be solved
		* @param _convergenceEpsilon minimum change in residual / energy under which the algorithm terminates
		* @param _perfData vector of performance data (residuals after every microiteration)
		* @returns the residual @f$|x-b|@f$ of the final @a _x
		*/
		double operator()(TTTensor &_x, const TTTensor &_b, value_t _convergenceEpsilon, PerformanceData &_perfData = NoPerfData) const {
			return solve(nullptr, _x, _b, numSteps, _convergenceEpsilon, _perfData);
		}
		
		/**
		* call to minimze @f$ \|x - b\|^2 @f$ for @f$ x @f$
		* @param[in,out] _x in: initial guess, out: solution as found by the algorithm
		* @param _b right-hand side of the equation to be solved
		* @param _numSteps maximum number of half-sweeps to perform
		* @param _perfData vector of performance data (residuals after every microiteration)
		* @returns the residual @f$|x-b|@f$ of the final @a _x
		*/
		double operator()(TTTensor &_x, const TTTensor &_b, size_t _numSteps, PerformanceData &_perfData = NoPerfData) const {
			return solve(nullptr, _x, _b, _numSteps, convergenceEpsilon, _perfData);
		}
		
		double operator()(TTTensor &_x, const TTTensor &_b, PerformanceData &_perfData = NoPerfData) const {
			return solve(nullptr, _x, _b, numSteps, convergenceEpsilon, _perfData);
		}
	};
	
	/// default variant of the steepest descent algorithm using the lapack solver
	extern const GeometricCGVariant GeometricCG;
}

