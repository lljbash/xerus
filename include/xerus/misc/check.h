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
* @brief Header file for CHECK and REQUIRE macros.
*/

#pragma once

#include "codeCoverage.h"

/**
 * @def XERUS_CHECK(condition, level, message)
 * @brief Checks whether @a condition is true and calls XERUS_LOG(level, message) otherwise.
 * @details The check is omitted if level is not logged anyways.
 */

/**
 * @def XERUS_REQUIRE(condition, message)
 * @brief Checks whether @a condition is true. logs an error otherwise via XERUS_LOG(error, message).
 */

/**
 * @def XERUS_INTERNAL_CHECK(condition, message)
 * @brief Checks whether @a condition is true. logs a critical error otherwise via XERUS_LOG(fatal, message). and prints information on how to report a bug
 */

/**
 * @def XERUS_IF_CHECK(expression)
 * @brief Executes @a expression only if the compilation was without XERUS_DISABLE_RUNTIME_CHECKS set.
 */


#ifndef XERUS_DISABLE_RUNTIME_CHECKS
	#include "namedLogger.h"
	#include "callStack.h"
	#include "containerOutput.h"

	#define XERUS_CHECK(condition, level, message) XERUS_REQUIRE_TEST; if(XERUS_IS_LOGGING(level) && !(condition)) { XERUS_LOG(level, #condition " failed msg: " << message); } else void(0)

	#define XERUS_REQUIRE(condition, message) XERUS_CHECK(condition, error, message)
	
	#define XERUS_INTERNAL_CHECK(condition, message) \
		if(!(condition)) { \
			::std::cerr << "\n"\
				"########################################################################\n"\
				"###                    AN INTERNAL ERROR OCCURED!                    ###\n"\
				"###  Please send the following information to contact@libxerus.org ! ###\n"\
				"########################################################################\n"\
				"xerus version: " << ::xerus::VERSION_MAJOR << '.' << ::xerus::VERSION_MINOR << '.' << ::xerus::VERSION_REVISION << '-' << ::xerus::VERSION_COMMIT << "\n" \
				"at: " __FILE__ " : " XERUS_STRINGIFY(__LINE__) "\n" \
				"message: " #condition " failed: " << message << '\n' \
				<< "callstack: \n" << ::xerus::misc::get_call_stack() << "\n" \
				"########################################################################\n"\
				"###                           thank you :-)                          ###\n"\
				"########################################################################" << std::endl; \
			XERUS_LOG(critical, #condition " failed msg: " << message); \
		} else void(0)
	
	#define XERUS_IF_CHECK(expression) expression
	
	#define XERUS_IF_NO_CHECK(expression)
#else
	#define XERUS_CHECK(condition, level, message) XERUS_REQUIRE_TEST; void(0)
	
	#define XERUS_REQUIRE(condition, message) XERUS_REQUIRE_TEST; void(0)
	
	#define XERUS_INTERNAL_CHECK(condition, msg) XERUS_REQUIRE_TEST; void(0)
	
	#define XERUS_IF_CHECK(expression)
	
	#define XERUS_IF_NO_CHECK(expression) expression
#endif
