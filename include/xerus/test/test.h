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
 * @brief Header file for unit test and code-coverage helper classes.
 */

#pragma once


/**
 * @def XERUS_REQUIRE_TEST
 * @brief Marked position for the code-coverage test. Any not passed XERUS_REQUIRE_TEST macro will result in a warning.
 * XERUS_REQUIRE_TEST is implied by any XERUS_REQUIRE(...) macro if test coverage is enabled.
 */

#include <map>
#include <string>
#include <functional>

#include "../misc/namedLogger.h"

#ifdef XERUS_TEST_COVERAGE
	#define XERUS_REQUIRE_TEST \
		do { \
			static const char * xerus_test_fname = __PRETTY_FUNCTION__;\
			struct xerus_test_a{ static void rt() {\
				xerus::misc::internal::RequiredTest::register_test(xerus_test_fname, __FILE__, __LINE__);\
			} };\
			using xerus_test_t = void (*)();\
			static xerus_test_t xerus_test_rtp __attribute__((section(".init_array"))) = &xerus_test_a::rt; \
			(void)xerus_test_rtp; \
			xerus::misc::internal::RequiredTest::increase_counter(xerus_test_fname, __FILE__, __LINE__); \
		} while(false)
#else
	#define XERUS_REQUIRE_TEST (void)0
#endif

#define XERUS_PRINTCHECK std::cout << u8"\033[1;32m\u2713 \033[0m" << std::flush
#define XERUS_PRINTFAIL  std::cout << u8"\033[1;31m\u2717 \033[0m" << std::flush
#define XERUS_PRINTSKIP  std::cout << u8"\033[1;33m\u2297 \033[0m" << std::flush

#ifndef XERUS_DISABLE_RUNTIME_CHECKS
	#define FAILTEST(test) \
		{\
			bool failtestFailed = false; \
			xerus::misc::internal::logFilePrefix = "failtest/"; xerus::misc::internal::silenced = true; \
			try { test; } catch (...) {failtestFailed = true; XERUS_PRINTCHECK;} \
			xerus::misc::internal::logFilePrefix.clear();  xerus::misc::internal::silenced = false;  \
			if(!failtestFailed) { XERUS_PRINTFAIL; XERUS_LOG(warning, #test << " returned without error"); ::xerus::misc::UnitTest::passed = false; } \
		}\
		void(0)
#else
	#define FAILTEST(test) XERUS_LOG(warning, "Failtest is not useful with flag XERUS_DISABLE_RUNTIME_CHECKS")
#endif


///@brief: A single test of the given condition. To be used inside a misc::UnitTest.
#define TEST(...) if (!(__VA_ARGS__)) {XERUS_PRINTFAIL; XERUS_LOG(warning, #__VA_ARGS__ << " failed"); ::xerus::misc::UnitTest::passed = false;} else {XERUS_PRINTCHECK;} void(0)

///@brief: A single test of the given condition @a cond. To be used inside a misc::UnitTest. Prints the given error message on failure.
#define MTEST(cond, ...) if (!(cond)) {XERUS_PRINTFAIL; XERUS_LOG(warning, #cond << " failed. Msg: " << __VA_ARGS__); ::xerus::misc::UnitTest::passed = false;} else {XERUS_PRINTCHECK;} void(0)

///@brief: A single skippable test of the given condition @a cond. To be used inside a misc::UnitTest.
#define TEST_SKIPPABLE(...) if (!(__VA_ARGS__)) {XERUS_PRINTSKIP; XERUS_LOG(debug, #__VA_ARGS__ << " failed"); } else {XERUS_PRINTCHECK;} void(0)

///@brief: A single skippable test of the given condition @a cond. To be used inside a misc::UnitTest. Prints the given debug message on failure.
#define MTEST_SKIPPABLE(cond, ...) if (!(cond)) {XERUS_PRINTSKIP; XERUS_LOG(debug, "Skipped test:" << #cond << " failed. Msg: " << __VA_ARGS__);} else {XERUS_PRINTCHECK;} void(0)


namespace xerus { namespace misc {
	#ifdef XERUS_UNITTEST
		struct UnitTest final {
			// order of contruction of global objects is random so the first one has to create the map
			static std::map<std::string, std::map<std::string, std::function<void()>>> *tests;
			static bool passed;
			
			UnitTest(std::string _group, std::string _name, std::function<void()> _f);
		};
	#endif

	namespace internal {
	#ifdef XERUS_TEST_COVERAGE
		struct RequiredTest {
			struct Identifier {
				std::string functionName;
				std::string filename;
				size_t lineNumber;
				Identifier(std::string _func, std::string _file, size_t _line);
				bool operator<(const Identifier &_rhs) const;
			};
			
			// order of construction of global objects is random so the first one has to create the map
			static std::map<Identifier, size_t> *tests;
			
			static void register_test(std::string _functionName, std::string _fileName, size_t _lineNb);
			
			static void increase_counter(std::string _functionName, std::string _fileName, size_t _lineNb);
		};
	#endif
}}}
