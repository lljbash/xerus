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
#include<fstream>

#include "../../include/xerus/test/test.h"
using namespace xerus;


// UNIT_TEST(SteepestDescent, random_operator,
// 	std::mt19937_64 rnd(rd());
// 	std::normal_distribution<value_t> dist (0.0, 1.0);
// 	
// 	const size_t d = 10;
// 	const std::vector<size_t> stateDims(d, 5);
// 	const std::vector<size_t> operatorDims(2*d, 5);
// 	
// 	Index i,j,k;
// 	PerformanceData perfdata;
// 	
// 	HOSVDRetraction svdRet(2ul);
// 	SteepestDescentVariant hosvdGrad(svdRet);
// // 	CGVariant hosvdGrad(svdRet);
// 	hosvdGrad.printProgress = true;
// 	hosvdGrad.assumeSymmetricPositiveDefiniteOperator = true;
// 	
// 	SteepestDescentVariant alsGrad(ALSRetraction);
// // 	CGVariant alsGrad(ALSRetraction);
// 	alsGrad.printProgress = true;
// 	alsGrad.assumeSymmetricPositiveDefiniteOperator = true;
// 	
// 	SteepestDescentVariant submanGrad(SubmanifoldRetraction);
// // 	CGVariant submanGrad(SubmanifoldRetraction);
// 	submanGrad.printProgress = true;
// 	submanGrad.assumeSymmetricPositiveDefiniteOperator = true;
// 	
// 	ALSVariant ALSb(ALS);
// 	ALSb.printProgress = true;
// 	ALSb.useResidualForEndCriterion = true;
// 	
// 	PerformanceData::Histogram hosvdHist(0.9), gradalsHist(0.9), submaniHist(0.9), alsHist(0.9);
// 	
// 	
// 	
// 	while (true) {
// 		TTOperator A = TTOperator::random(operatorDims, std::vector<size_t>(d-1, 2), dist);
// 		A(i/2, j/2) = A(i/2, k/2) * A(j/2, k/2);
// 		TTTensor b = TTTensor::random(stateDims, std::vector<size_t>(d-1,4), dist);
// 		b(i&0) = A(i/2,j/2)*b(j&0);
// 		TTTensor initX = TTTensor::random(stateDims, std::vector<size_t>(d-1,4), dist);
// 		TTTensor x(initX);
// 		
// 		try {
// 			perfdata.reset();
// 			hosvdGrad(A,x,b,4000ul,perfdata);
// 			std::cout << "HOSVD: " << perfdata.data.size() << " " << perfdata.data.back().residual << " " << perfdata.data.back().elapsedTime << std::endl;
// 			hosvdHist += perfdata.get_histogram(0.9);
// 			hosvdHist.dump_to_file("hosvd.dat");
// 			
// 			perfdata.reset();
// 			x = initX;
// 			alsGrad(A,x,b,4000ul, perfdata);
// 			std::cout << "grad(ALS): " << perfdata.data.size()  << " " << perfdata.data.back().residual << " " << perfdata.data.back().elapsedTime << std::endl;
// 			gradalsHist += perfdata.get_histogram(0.9);
// 			gradalsHist.dump_to_file("gradals.dat");
// 			
// 			perfdata.reset();
// 			x = initX;
// 			submanGrad(A,x,b,4000ul, perfdata);
// 			std::cout << "Submanifold: " << perfdata.data.size()  << " " << perfdata.data.back().residual << " " << perfdata.data.back().elapsedTime << std::endl;
// 			submaniHist += perfdata.get_histogram(0.9);
// 			submaniHist.dump_to_file("submani.dat");
// 			perfdata.dump_to_file("submaniPlot.dat");
// 			
// 			perfdata.reset();
// 			x = initX;
// 			ALSb(A,x,b,50ul, perfdata);
// 			std::cout << "ALS: " << perfdata.data.size()  << " " << perfdata.data.back().residual << " " << perfdata.data.back().elapsedTime << std::endl;
// 			alsHist += perfdata.get_histogram(0.9);
// 			alsHist.dump_to_file("als.dat");
// 		} catch (xerus::misc::generic_error &) { }
// 	}
// 	
// )
