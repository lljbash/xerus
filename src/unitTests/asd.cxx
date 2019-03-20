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

#include <common.hxx>


static misc::UnitTest trasd_frr("TRASD", "Fixed_Rank_Recovery", [](){
	const std::vector<size_t> orders{2, 3, 5};
	const size_t runs = 6;
	const size_t successThreshold = 4;
	
	auto myTRASD = TRASD;
	
	myTRASD.maxIterations = 500;
	myTRASD.targetRelativeResidual = 1e-6;
	
	for(const auto d : orders) {
		size_t residualSucc = 0, ErrorSucc = 0;
		
		for(size_t i = 0; i < runs; ++i) {
			const auto dimensions = random_dimensions(d, 1, 8);
			
			if(misc::product(dimensions) == 1) {continue;} // TODO Check why this doesn't work.
			
			const auto ranks = random_low_tt_ranks(dimensions, 1, 6);

			const auto target = TTTensor::random(dimensions, ranks);
			const double targetNorm = frob_norm(target);
			const size_t targetDofs = target.datasize();
			
			// This should allow succesfull reconstruction in almost all cases.
			const size_t N = 2*targetDofs+misc::product(target.dimensions)/25;
			
			const auto measurments = RankOneMeasurementSet::random(N, target);
			
			
			TTTensor solution = TTTensor::random(target.dimensions, ranks);
			myTRASD(solution, measurments);
			const double error = frob_norm(target-solution)/targetNorm;
			const double residual = measurments.test(solution);
			if(residual < 1e-6) { residualSucc++; }
			if(error < 1e-4) { ErrorSucc++; }
		}
		
		MTEST(residualSucc >= successThreshold, "Only " << residualSucc << " of " << runs << " were succesfull in terms of residual.");
		MTEST(ErrorSucc >= successThreshold, "Only " << ErrorSucc << " of " << runs << " were succesfull in terms of error.");
	}
});


static misc::UnitTest trasd_rar("TRASD", "Rank_Adaptive_Recovery", [](){
	const std::vector<size_t> orders{2, 3, 4};
	const size_t runs = 6;
	const size_t successThreshold = 4;
	
	auto myTRASD = TRASD;
	
	myTRASD.maxIterations = 500;
	myTRASD.targetRelativeResidual = 1e-6;
	
	for(const auto d : orders) {
		size_t residualSucc = 0, ErrorSucc = 0;
		
		for(size_t i = 0; i < runs; ++i) {
			const auto dimensions = random_dimensions(d, 1, 7);
			
			if(misc::product(dimensions) == 1) {continue;} // TODO Check why this doesn't work.
			
			const auto ranks = random_low_tt_ranks(dimensions, 1, 5);

			const auto target = TTTensor::random(dimensions, ranks);
			const double targetNorm = frob_norm(target);
			const size_t targetDofs = target.datasize();
			
			// This should allow succesfull reconstruction in almost all cases.
			const size_t N = 4*targetDofs+misc::product(target.dimensions)/4;
			
			const auto measurments = RankOneMeasurementSet::random(N, target);
			
			
			TTTensor solution = TTTensor::random(target.dimensions, std::vector<size_t>(d-1, 1));
			myTRASD(solution, measurments, std::vector<size_t>(d-1, 10));
			const double error = frob_norm(target-solution)/targetNorm;
			const double residual = measurments.test(solution);
			if(residual < 1e-4) { residualSucc++; }
			if(error < 1e-4) { ErrorSucc++; }
		}
		
		MTEST(residualSucc >= successThreshold, "Only " << residualSucc << " of " << runs << " were succesfull in terms of residual.");
		MTEST(ErrorSucc >= successThreshold, "Only " << ErrorSucc << " of " << runs << " were succesfull in terms of error.");
	}
});


// static misc::UnitTest trasd_frc("TRASD", "Fixed_Rank_Completion", [](){
// 	const std::vector<size_t> orders{2, 3, 5};
// 	const size_t runs = 100;
// 	const size_t successThreshold = 100;
// 	
// 	auto myTRASD = TRASD;
// 	
// 	myTRASD.maxIterations = 500;
// 	myTRASD.targetRelativeResidual = 1e-6;
// 	
// 	for(const auto d : orders) {
// 		size_t residualSucc = 0, ErrorSucc = 0;
// 		
// 		for(size_t i = 0; i < runs; ++i) {
// 			const auto dimensions = random_dimensions(d, 1, 8);
// 			
// 			if(misc::product(dimensions) == 1) {continue;} // TODO Check why this doesn't work.
// 			
// 			const auto ranks = random_low_tt_ranks(dimensions, 1, 6);
// 
// 			const auto target = TTTensor::random(dimensions, ranks);
// 			const double targetNorm = frob_norm(target);
// 			const size_t targetDofs = target.datasize();
// 			
// 			// This should allow succesfull reconstruction in almost all cases.
// 			const size_t N = std::min(misc::product(dimensions), 5*targetDofs+misc::product(target.dimensions)/25);
// 			
// 			const auto completionMeasurments = RankOneMeasurementSet(SinglePointMeasurementSet::random(N, target), target.dimensions);
// 			
// 			
// 			TTTensor solution = TTTensor::random(target.dimensions, ranks);
// 			myTRASD(solution, completionMeasurments);
// 			const double error = frob_norm(target-solution)/targetNorm;
// 			const double residual = completionMeasurments.test(solution);
// 			if(residual < 1e-6) { residualSucc++; }
// 			if(error < 1e-4) { ErrorSucc++; }
// 		}
// 		
// 		MTEST(residualSucc >= successThreshold, "Only " << residualSucc << " of " << runs << " were succesfull in terms of residual.");
// 		MTEST(ErrorSucc >= successThreshold, "Only " << ErrorSucc << " of " << runs << " were succesfull in terms of error.");
// 	}
// });


