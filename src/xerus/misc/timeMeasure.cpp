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
 * @brief Implementation of the TimeMeasure class.
 */

#include <xerus/misc/timeMeasure.h>
#include <chrono>

namespace xerus {
        namespace misc {
        size_t uTime() {
            return static_cast<size_t>(std::chrono::duration_cast<std::chrono::microseconds>
                (std::chrono::system_clock::now().time_since_epoch()).count());
        }

        size_t mTime() {
            return static_cast<size_t>(std::chrono::duration_cast<std::chrono::milliseconds>
                (std::chrono::system_clock::now().time_since_epoch()).count());
        }
    } // namespace misc
} // namespace xerus
