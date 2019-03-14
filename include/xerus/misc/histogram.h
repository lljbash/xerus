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
 * @brief Header file for the histogram classes.
 */

#pragma once 

#include <string>
#include <fstream>
#include <map>

namespace xerus { 
	namespace misc {

		/**
		* @brief A logarithmic histogram, i.e. the size of all buckets is given by a constant factor [x - x*base)
		*/
		class LogHistogram {
		public:
			double base;
			std::map<int, size_t> buckets;
			size_t totalCount;
			
			explicit LogHistogram(const double _base);
			
			LogHistogram &operator+=(const LogHistogram &_other);
			void add(double _value, size_t _count = 1);
			
			static LogHistogram read_from_file(const std::string &_fileName);
			void dump_to_file(const std::string &_fileName) const;
		};
	}
}
