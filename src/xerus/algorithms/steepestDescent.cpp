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
 * @brief Implementation of the steepest descent algorithms.
 */

#include <xerus/algorithms/steepestDescent.h>
#include <xerus/algorithms/als.h>
#include <xerus/basic.h>
#include <xerus/index.h>
#include <xerus/indexedTensorList.h>
#include <xerus/tensorNetwork.h>
 
#include <xerus/indexedTensorMoveable.h>
#include <xerus/indexedTensor_tensor_factorisations.h>

namespace xerus {
	
	void line_search(TTTensor &_x, value_t &_alpha, const TTTangentVector &_direction, value_t _derivative, value_t &_residual,
					 std::function<void(TTTensor &, const TTTangentVector &)> _retraction,
					 std::function<value_t()> _calculate_residual,
					 value_t _changeInAlpha)
	{
		value_t dirNorm = _direction.frob_norm();
		value_t currAlpha = _alpha / _changeInAlpha;
		TTTensor oldX(_x);
		_retraction(_x, currAlpha/dirNorm * _direction);
		value_t bestResidual = _calculate_residual();
		value_t bestAlpha = currAlpha;
		TTTensor bestX = _x;
		
		do {
			currAlpha *= _changeInAlpha;
			_x = oldX;
			_retraction(_x, currAlpha/dirNorm * _direction);
			value_t newResidual = _calculate_residual();
// 			LOG(line_search, currAlpha << " -> " << newResidual);
			if (newResidual < bestResidual) {
				bestResidual = newResidual;
				bestAlpha = currAlpha;
				bestX = _x;
			} else {
				break;
			}
		} while(true);
		
		_x = std::move(bestX);
		_alpha = bestAlpha;
		
		// armijo backtracking
		const value_t min_decrease = 1e-4;
// 		LOG(armijo, bestResidual << " > " << _residual << " - " << min_decrease << " * " << _alpha << " * " << _derivative << " / " << dirNorm);
		while (_alpha > 1e-16 && bestResidual > _residual - min_decrease * _alpha/dirNorm * _derivative) {
// 			LOG(armijo, _alpha << " " << bestResidual);
			_alpha *= _changeInAlpha;
			_x = oldX;
			_retraction(_x, _alpha/dirNorm * _direction);
			bestResidual = _calculate_residual();
		}
		
		_residual = bestResidual;
	}
	
	
	value_t SteepestDescentVariant::solve(const TTOperator *_Ap, TTTensor &_x, const TTTensor &_b, size_t _numSteps, value_t _convergenceEpsilon, PerformanceData &_perfData) const {
		const TTOperator &_A = *_Ap;
		static const Index i,j;
		size_t stepCount=0;
		TTTensor residual;
		value_t lastResidual=1e100;
		value_t currResidual=1e100;
// 		value_t normB = frob_norm(_b);
		
		
		if (_Ap != nullptr) {
			_perfData << "Steepest Descent for ||A*x - b||^2, x.dimensions: " << _x.dimensions << '\n'
					<< "A.ranks: " << _A.ranks() << '\n';
			if (assumeSymmetricPositiveDefiniteOperator) {
				_perfData << " with symmetric positive definite Operator A\n";
			}
		} else {
			_perfData << "Steepest Descent for ||x - b||^2, x.dimensions: " << _x.dimensions << '\n';
		}
		_perfData << "x.ranks: " << _x.ranks() << '\n'
					<< "b.ranks: " << _b.ranks() << '\n'
					<< "maximum number of steps: " << _numSteps << '\n'
					<< "convergence epsilon: " << _convergenceEpsilon << '\n';
		_perfData.start();
		
		auto updateResidual = [&]() {
			if (_Ap != nullptr) {
				residual(i&0) = _b(i&0) - _A(i/2,j/2)*_x(j&0);
			} else {
				residual = _b - _x;
			}
			currResidual = frob_norm(residual);
		};
		
		auto updatePerfdata = [&]() {
			_perfData.add(currResidual, _x);
		};
		updateResidual();
		updatePerfdata();
		
		TTTensor y, Ay;
		value_t alpha = 1;
		while ((_numSteps == 0 || stepCount < _numSteps)
			   && currResidual > _convergenceEpsilon
			   && std::abs(lastResidual-currResidual) > _convergenceEpsilon
			   && std::abs(1-currResidual/lastResidual) > _convergenceEpsilon) 
		{
			stepCount += 1;
			
			if (_Ap != nullptr) {
				if (assumeSymmetricPositiveDefiniteOperator) { XERUS_REQUIRE_TEST;
					// search direction: y = b-Ax
					y = residual;
					if (preconditioner != nullptr) {
						y(j&0) = (*preconditioner)(j/2,i/2) * y(i&0);
					}
					// direction of change A*y
// 					Ay(i&0) = _A(i/2,j/2) * y(j&0);
					// "optimal" stepsize alpha = <y,y>/<y,Ay>
// 					alpha = misc::sqr(frob_norm(y)) / value_t(y(i&0)*Ay(i&0));
				} else { XERUS_REQUIRE_TEST;
					// search direction: y = A^T(b-Ax)
					y(i&0) = _A(j/2,i/2) * residual(j&0);
					if (preconditioner != nullptr) {
						y(j&0) = (*preconditioner)(j/2,i/2) * y(i&0);
					}
					// direction of change A*y
// 					Ay(i&0) = _A(i/2,j/2) * y(j&0);
					// "optimal" stepsize alpha = <y,y>/<Ay,Ay>
// 					alpha = misc::sqr(frob_norm(y)) / misc::sqr(frob_norm(Ay));
				}
			} else {
				y = residual;
			}
			
			TTTensor oldX(_x);
			alpha *= 2;
			retraction(_x, y * alpha);
			lastResidual = currResidual;
			updateResidual();
			// armijo backtracking
			while (alpha > 1e-30 && lastResidual < currResidual){// - 1e-4 * alpha * value_t(residual(i&0) * y(i&0))) {
// 				LOG(huch, alpha << " " << currResidual); 
				alpha /= 2;
				_x = oldX;
				retraction(_x, y * alpha);
				updateResidual();
			}
			
			updatePerfdata();
		}
		
		return currResidual;
	}
	
	const SteepestDescentVariant SteepestDescent(0, 1e-8, false, SubmanifoldRetractionII);
} // namespace xerus
