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
 * @brief Implementation of the CG algorithm.
 */

#include <xerus/algorithms/cg.h>
#include <xerus/algorithms/als.h>
#include <xerus/algorithms/steepestDescent.h>
#include <xerus/basic.h>
#include <xerus/index.h>
#include <xerus/indexedTensorList.h>
#include <xerus/tensorNetwork.h>
 
#include <xerus/indexedTensorMoveable.h>
#include <xerus/indexedTensor_tensor_factorisations.h>

namespace xerus {
	
	value_t GeometricCGVariant::solve(const TTOperator *_Ap, TTTensor &_x, const TTTensor &_b, size_t _numSteps, value_t _convergenceEpsilon, PerformanceData &_perfData) const {
		const TTOperator &_A = *_Ap;
		static const Index i,j;
		size_t stepCount=0;
		TTTensor residual;
		TTTangentVector gradient;
		value_t gradientNorm = 1.0;
		value_t lastResidual=1e100;
		value_t currResidual=1e100;
		value_t normB = frob_norm(_b);
		
		if (_Ap != nullptr) {
			_perfData << "Conjugated Gradients for ||A*x - b||^2, x.dimensions: " << _x.dimensions << '\n'
					<< "A.ranks: " << _A.ranks() << '\n';
			if (assumeSymmetricPositiveDefiniteOperator) {
				_perfData << " with symmetric positive definite Operator A\n";
			}
		} else {
			_perfData << "Conjugated Gradients for ||x - b||^2, x.dimensions: " << _x.dimensions << '\n';
		}
		_perfData << "x.ranks: " << _x.ranks() << '\n'
					<< "b.ranks: " << _b.ranks() << '\n'
					<< "maximum number of steps: " << _numSteps << '\n'
					<< "convergence epsilon: " << _convergenceEpsilon << '\n';
		_perfData.start();
		
		auto calculateResidual = [&]()->value_t {
			if (_Ap != nullptr) {
				residual(i&0) = _b(i&0) - _A(i/2,j/2)*_x(j&0);
			} else {
				residual = _b - _x;
			}
			return frob_norm(residual);//normB;
		};
		auto updateGradient = [&]() {
			if (assumeSymmetricPositiveDefiniteOperator || (_Ap == nullptr)) {
				gradient = TTTangentVector(_x, residual);
			} else {
				TTTensor grad;
				grad(i&0) = (*_Ap)(j/2,i/2) * residual(j&0); // grad = A^T * (b - Ax)
				gradient = TTTangentVector(_x, grad);
			}
			gradientNorm = gradient.frob_norm();
		};
		
		currResidual = calculateResidual();
		_perfData.add(stepCount, currResidual, _x);
		
		updateGradient();
		TTTangentVector direction = gradient;
		value_t alpha = 1;
		while ((_numSteps == 0 || stepCount < _numSteps)
			&& currResidual/normB > _convergenceEpsilon
			&& std::abs(lastResidual-currResidual)/normB > _convergenceEpsilon
			&& std::abs(1-currResidual/lastResidual)/normB > _convergenceEpsilon) 
		{
			stepCount += 1;
			size_t stepFlags = 0;
			
			// check the derivative along the current direction
			value_t derivative = gradient.scalar_product(direction) / direction.frob_norm();
			
			// if movement in the given direction would increase the residual rather than decrease it, perform one steepest descent step instead
			if (derivative <= 0) {
				direction = gradient;
				derivative = gradient.frob_norm();
				alpha = 1;
				stepFlags |= 1;
			}
			
			line_search(_x, alpha, direction, derivative, currResidual, retraction, calculateResidual, 0.8);
			
			_perfData.add(stepCount, currResidual, _x, stepFlags);
			
// 			direction(i&0) = residual(i&0) + beta * direction(i&0);
			TTTangentVector oldDirection(direction);
			vectorTransport(_x, oldDirection);
			value_t oldGradNorm = gradientNorm;
			updateGradient();
			
			double beta = gradientNorm / oldGradNorm ;// Fletcher-Reeves update
			direction = gradient;
			direction += oldDirection * beta;
		}
		
		return currResidual;
	}
	
	const GeometricCGVariant GeometricCG(0, 1e-8, false, SubmanifoldRetractionI, ProjectiveVectorTransport);
} // namespace xerus
