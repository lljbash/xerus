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
 * @brief Implementation of the non-trivial non-template functions in the missingFunctions header.
 */

#include <xerus/misc/missingFunctions.h>
#include <xerus/misc/check.h>
#include <xerus/misc/internal.h>

namespace xerus {
    namespace misc {
		std::string exec(const std::string & _cmd) {
			FILE* pipe = popen(_cmd.c_str(), "r");
			REQUIRE(pipe, "could not start " << _cmd);
			char buffer[128];
			std::string result = "";
			while(feof(pipe) == 0) {
				if(fgets(buffer, 128, pipe) != nullptr) {
					result += buffer;
				}
			}
			pclose(pipe);
			return result;
		}
		
		void exec(const std::string & _cmd, const std::string &_stdin) {
			FILE* pipe = popen(_cmd.c_str(), "w");
			REQUIRE(pipe, "could not start " << _cmd);
			fputs(_stdin.c_str(), pipe);
			pclose(pipe);
		}
	} // namespace misc
} // namespace xerus
