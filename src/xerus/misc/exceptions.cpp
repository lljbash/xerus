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
 * @brief Implementation of the generic_error class.
 */

#include <xerus/misc/exceptions.h>

namespace xerus {
    namespace misc {
        generic_error::generic_error() {}
            
        generic_error::generic_error(const generic_error &_other) noexcept
            : error_info(_other.error_info) { }
        
        const char* generic_error::what() const noexcept {
            return error_info.c_str();
        }
        
    } // namespace misc
} // namespace xerus
