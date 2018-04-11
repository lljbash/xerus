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


#include<xerus.h>

#include "../../include/xerus/test/test.h"
using namespace xerus;

// UNIT_TEST(algorithms, cg,
// 	std::uniform_real_distribution<value_t> dist (0.0, 1.0);
//     
//     Index i,j,k;
//     
//     TTTensor X = TTTensor::random({10,10,10,10}, {4,4,4}, dist);
// 	TTTensor B = TTTensor::random({10,10,10,10}, {2,2,2}, dist);
// 	TTOperator A = TTOperator::random({10,10,10,10,10,10,10,10}, {2,2,2}, dist);
// 	
// // 	A(i/2,j/2) = A(k/2,i/2) * A(k/2,j/2);
// // 	B(i&0) = A(i/2,j/2) * B(j&0);
// 
// 	PerformanceData pdata(true);
// 	GeometricCGVariant cg(GeometricCG);
// // 	cg.assumeSymmetricPositiveDefiniteOperator = true;
// 	cg.retraction = ALSRetractionI;
// // 	cg.restartInterval = 20;
// 	cg.convergenceEpsilon = 0;
// 	
// 	cg(A, X, B, pdata);
// 	
// 	SteepestDescentVariant alsGrad(ALSRetractionII);
// 	alsGrad.printProgress = true;
// 	alsGrad.assumeSymmetricPositiveDefiniteOperator = true;
// 	
// 	alsGrad(A, X, B);
// )

