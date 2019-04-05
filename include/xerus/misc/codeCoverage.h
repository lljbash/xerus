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
 * @brief Header file for the code-coverage functions.
 */

#pragma once

#include <string>
#include <map>
#include <unordered_set>

#ifdef XERUS_TEST_COVERAGE

#define XERUS_S1(x) #x
#define XERUS_S2(x) XERUS_S1(x)
#define XERUS_LOCATION __FILE__ ":" XERUS_S2(__LINE__)
#define XERUS_LOCATION_MARKED "@" XERUS_LOCATION "@"

#define XERUS_DO_PRAGMA(x) _Pragma (#x)
#define XERUS_CC_MARK(msg) XERUS_DO_PRAGMA(message ("REQUIRE_TEST @" msg "@"))

/**
 * @def XERUS_REQUIRE_TEST
 * @brief Marked position for the code-coverage test. Any not passed XERUS_REQUIRE_TEST macro will result in a warning.
 * XERUS_REQUIRE_TEST is implied by any XERUS_REQUIRE(...) macro if test coverage is enabled.
 */
#define XERUS_REQUIRE_TEST \
	do { \
		static const char * const xerusCCLocalFunctionLocation = XERUS_LOCATION;\
		static const char * const xerusCCLocalFunctionName = __PRETTY_FUNCTION__;\
		__asm__( \
			".pushsection .cc_loc, \"a\", @progbits" "\n" \
			".quad %c0" "\n" \
			".quad %c1" "\n" \
			".popsection" "\n" \
			: : "i"(&xerusCCLocalFunctionLocation), "i"(&xerusCCLocalFunctionName) \
		); \
		xerus::misc::CodeCoverage::covered(xerusCCLocalFunctionLocation, xerusCCLocalFunctionName); \
		\
		XERUS_CC_MARK(XERUS_LOCATION) \
	} while(false)

#else
	#define XERUS_REQUIRE_TEST (void)0
#endif

namespace xerus { namespace misc { namespace CodeCoverage {
	
	extern std::map<std::string, std::unordered_set<std::string>>* testsCovered;
	extern std::map<std::string, std::unordered_set<std::string>>* testsRequiredInit;
	
	void covered(const std::string& _location, const std::string& _identifier);
	
	void print_code_coverage();
}}}
