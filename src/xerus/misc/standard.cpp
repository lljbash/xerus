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
 * @brief compilation unit for the version constants
 */

#include <xerus/misc/standard.h>

#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)


namespace xerus {
	const std::string VERSION_FULL = STR(XERUS_VERSION);
	const int VERSION_MAJOR = XERUS_VERSION_MAJOR;
	const int VERSION_MINOR = XERUS_VERSION_MINOR;
	const int VERSION_REVISION = XERUS_VERSION_REVISION;
	const int VERSION_COMMIT = XERUS_VERSION_COMMIT;
}

