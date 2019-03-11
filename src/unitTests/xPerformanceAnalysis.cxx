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


#include<xerus.h>
#include "../../include/xerus/misc/internal.h"

#include "../../include/xerus/test/test.h"
using namespace xerus;

#ifdef XERUS_PERFORMANCE_ANALYSIS
	#ifdef XERUS_REPLACE_ALLOCATOR  
		static misc::UnitTest perfana("x_PerformanceAnalysis_x", "Analysis", [](){
			std::cout << misc::performanceAnalysis::get_analysis();
			LOG(Indices, "A total of " << Index().valueId << " indices were used (in this thread).");
			
			using xma = xerus::misc::AllocatorStorage;
			namespace xm = xerus::misc; 
			LOG(allocator, "");
			size_t totalStorage=0;
			for (size_t i=0; i<xma::NUM_BUCKETS; ++i) {
				if (xm::astore.allocCount[i] == 0 && xm::astore.currAlloc[i] == 0) continue;
				totalStorage += i * xma::BUCKET_SIZE * (size_t)xm::astore.maxAlloc[i];
				LOG(allocator, (i+1) * xma::BUCKET_SIZE-1 << " \tx\t " << xm::astore.allocCount[i] << "\tmax: " << xm::astore.maxAlloc[i] << '\t' << xm::astore.currAlloc[i] << '\t' << totalStorage);
			}
			LOG(storageNeeded, totalStorage << " storage used: " << misc::astore.pools.size()*xma::POOL_SIZE);
			LOG(index, sizeof(Index));
			LOG(node, sizeof(TensorNetwork::TensorNode));
			LOG(link, sizeof(TensorNetwork::Link));
			LOG(tn, sizeof(TensorNetwork));
			LOG(fulltensor, sizeof(Tensor));
			LOG(Tensor, sizeof(Tensor));
			LOG(tt, sizeof(TTTensor) << " | " << sizeof(TTOperator));
			LOG(measurement, sizeof(SinglePointMeasurment));
		});
	#else
		static misc::UnitTest perfana("x_PerformanceAnalysis_x", "Analysis", [](){
			std::cout << misc::performanceAnalysis::get_analysis();
			LOG(Indices, "A total of " << Index().valueId << " indices were used (in this thread).");
		});
	#endif
#endif
