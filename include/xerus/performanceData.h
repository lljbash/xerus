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
* @brief Header file for the PerformanceData class.
*/

#pragma once

#include <string>
#include <vector>

#include "misc/timeMeasure.h"
#include "misc/histogram.h"

#include "basic.h"
#include "tensorNetwork.h"
#include "forwardDeclarations.h"

namespace xerus {
	/// @brief Storage class for the performance data collected during an algorithm (typically iteration count, time and residual)
	class PerformanceData {
	public:
		struct DataPoint {
			size_t iteration;
			size_t elapsedTime;
			std::vector<double> residuals;
			double error;
			size_t dofs;
			size_t flags;
			
			DataPoint(const size_t _itrCount, const size_t _time, const std::vector<double>& _residual, const value_t _error, const size_t _dofs, const size_t _flags) 
				: iteration(_itrCount), elapsedTime(_time), residuals(_residual), error(_error), dofs(_dofs), flags(_flags) {}
		};
		
		const bool active;
		
		bool printProgress;
		
		using ErrorFunction = std::function<double(const TTTensor&)>;
		ErrorFunction errorFunction;
		
		size_t startTime;
		size_t stopTime;
		std::vector<DataPoint> data;
		
		std::string additionalInformation;
		
		
		explicit PerformanceData(const bool _printProgress = false, const bool _active = true);
		
		explicit PerformanceData(const ErrorFunction& _errorFunction, const bool _printProgress = false, const bool _active = true);
		
		void start();
		
		void stop_timer();
		
		void continue_timer();
		
		void reset();
		
		size_t get_elapsed_time() const;
		
		size_t get_runtime() const;
		
		void add(const double _residual, const TTTensor& _x, const size_t _flags = 0);
		
		void add(const std::vector<double>& _residuals, const TTTensor& _x, const size_t _flags = 0);
		
		void add(const size_t _itrCount, const double _residual, const TTTensor& _x, const size_t _flags = 0);
		
		void add(const size_t _itrCount, const std::vector<double>& _residuals, const TTTensor& _x, const size_t _flags);
		
		operator bool() const { return active; }
		
		/// @brief The pipe operator allows to add everything that can be converted to string to the additional information in the header. 
		template<class T>
		PerformanceData& operator<<(const T &_info) {
			using ::xerus::misc::operator<<;
			if (active) {
				additionalInformation += misc::to_string(_info);
				if(printProgress) {
					XERUS_LOG_SHORT(PerformanceData, _info);
				}
			}
			return *this;
		}
		
		void dump_to_file(const std::string &_fileName) const;
		
		misc::LogHistogram get_histogram(const value_t _base, bool _assumeConvergence = false) const;
	};

	extern PerformanceData NoPerfData;
}
