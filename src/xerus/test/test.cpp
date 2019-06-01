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
 * @brief Implementation of code coverage functions and the main routine for the unit-test executable.
 */

#include <xerus/test/test.h>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <signal.h>
#include <sstream>

#include <string.h> // for strsignal

#include <xerus/misc/standard.h>
#include <xerus/misc/exceptions.h>
#include <xerus/misc/stringUtilities.h>
#include <xerus/misc/fileUtilities.h>
#include <xerus/misc/random.h>
#include <xerus/misc/internal.h>


namespace xerus { namespace misc {
			
	std::map<std::string, std::map<std::string, std::function<void()>>> *xerus::misc::UnitTest::tests;
	bool xerus::misc::UnitTest::passed;
		
	UnitTest::UnitTest(std::string _group, std::string _name, std::function<void()> _f) {
		if (!tests) {
			tests = new std::map<std::string, std::map<std::string, std::function<void()>>>();
		}
		if (tests->count(_group) > 0 && (*tests)[_group].count(_name) > 0) {
			std::cerr << "ERROR: Unit test '" << _group << ":" << _name << "' defined multiple times!" << std::endl;
			while ((*tests)[_group].count(_name) > 0) {
				_name += "_";
			}
		}
		(*tests)[_group][_name] = _f;
	}
	
	namespace internal {
		
	bool test(const std::pair<std::string, std::function<void()>> &_t, uint64 _seed=0) {
		std::cout << "| " << _t.first << " starting: "  << std::flush;
		
		// initialize passed variable and reseed randomness
		::xerus::misc::UnitTest::passed = true;
		if (_seed == 0) {
#ifdef XERUS_TRULY_RANDOM_UNITTESTS
			_seed = ::xerus::misc::randomEngine();
#else
			_seed = 0xBAADF00D;
#endif
		}
		::xerus::misc::randomEngine.seed(_seed);
		
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		try {
			_t.second(); // executes the test
		} catch (const xerus::misc::generic_error &e) {
			std::cout << u8"\033[1;31m\u2717 \033[0m" << std::endl;
			std::cerr << "| Test has thrown an uncaught xerus::generic_error():" << std::endl;
			std::cerr << e.what() << std::endl;
			::xerus::misc::UnitTest::passed = false;
		} catch (const std::exception &e) {
			std::cout << u8"\033[1;31m\u2717 \033[0m" << std::endl;
			std::cerr << "| Test has thrown an uncaught std::exception:" << std::endl;
			std::cerr << e.what() << std::endl;
			::xerus::misc::UnitTest::passed = false;
		} catch (...) {
			std::cout << u8"\033[1;31m\u2717 \033[0m" << std::endl;
			std::cerr << "| Test has thrown an uncaught unknown exception..." << std::endl;
			::xerus::misc::UnitTest::passed = false;
		}
		std::chrono::microseconds::rep time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
		
		if (::xerus::misc::UnitTest::passed) { 
			std::cout << std::endl << "| " << _t.first << ":\033[1;32m passed!\033[0m (" << std::fixed << std::setprecision(3) << static_cast<double>(time)/1000.0 << " ms)\n| " << std::endl;
		} else {
			std::cout << std::endl << "| " << _t.first << ":\033[1;31m FAILED!\033[0m (" << std::fixed << std::setprecision(3) << static_cast<double>(time)/1000.0 << " ms, seed = " << std::hex << _seed << ")\n| " << std::endl;
		}
		
		return ::xerus::misc::UnitTest::passed;
	}

	void print_group_name(std::string _g) {
		int a = (77-static_cast<int>(_g.size()))/2; if (a<0) a=0;
		int b = 77-(77+static_cast<int>(_g.size()))/2; if (b<0) b=0;
		std::cout << "-------------------------------------------------------------------------------" << std::endl;
		std::cout << "|" << std::string(static_cast<size_t>(a), ' ') << "\033[1m" << _g << "\033[0m" << std::string(static_cast<size_t>(b), ' ')  << ' ' << std::endl;
		std::cout << "|" << std::endl;
		//std::cout << "-------------------------------------------------------------------------------" << std::endl;
	}

	void print_group_summary(const std::string& _g, const size_t _passes, const size_t _total) {
		std::stringstream ts;
		ts.str(""); ts.clear();
		ts << _g << " summary " << (_passes == _total?"\033[1;32m":"\033[1;31m") << _passes << " of " << _total << " passed\033[0m";
		int a = (77-(static_cast<int>(ts.str().size())-11))/2; if (a<0) a=0;
		int b = 77-(77+(static_cast<int>(ts.str().size())-11))/2; if (b<0) b=0;
		//std::cout << "-------------------------------------------------------------------------------" << std::endl;
		std::cout << "|" << std::endl;
		std::cout << "|" << std::string(static_cast<size_t>(a), ' ') << ts.str() << std::string(static_cast<size_t>(b), ' ')  << ' ' << std::endl;
		std::cout << "-------------------------------------------------------------------------------" << std::endl;
	}

	__attribute__((noreturn)) void catch_signals(int _sig)  {
		XERUS_THROW(xerus::misc::generic_error() << "signal " << _sig << " = " << strsignal(_sig) << "callstack:\n" << xerus::misc::get_call_stack());
	}
}}} // namespaces

#undef main
int main(int argc, char* argv[]) {
// 	signal(SIGINT, xerus::misc::internal::catch_signals); // users ctrl+c should actually terminate the program
// 	signal(SIGTERM, xerus::misc::internal::catch_signals);
// 	signal(SIGHUP, xerus::misc::internal::catch_signals);
	//signal(SIGABRT,xerus::misc::internal::catch_signals); // common source of abort is "double free or corurption" in which case we cannot continue
	signal(SIGFPE,xerus::misc::internal::catch_signals);
	signal(SIGILL,xerus::misc::internal::catch_signals);
	signal(SIGSEGV,xerus::misc::internal::catch_signals);
	
	//Calculate complete time
	std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();
	
	std::cout << "###############################################################################\n";
	std::cout << "#                                  Unit Tests                                 #\n";
	std::cout << "#                   Xerus version: " << std::setw(42) << std::left << xerus::VERSION_FULL << " #\n";
	std::cout << "###############################################################################" << std::endl;
	// No unittests defined (ie. the map tests does not exist!)
	if (!xerus::misc::UnitTest::tests) {
		std::cout << "No unittests defined.\n";
		std::cout << "Create xerus::misc::UnitTest objects to define unittests inside the sourcecode." << std::endl;
		return 0;
	}
	
	if (argc < 2) {
		std::cout << "usage:\n";
		std::cout << "  " << xerus::misc::explode(argv[0],'/').back() << " [groupname] ...\n";
		std::cout << "  " << xerus::misc::explode(argv[0],'/').back() << " [groupname]:[testname] ...\n";
		std::cout << "  " << xerus::misc::explode(argv[0],'/').back() << " [groupname]:[testname]:[seed] ...\n";
		std::cout << "  " << xerus::misc::explode(argv[0],'/').back() << " all\n\n";
		std::cout << "available groups:\n";
		for (const auto &p : *xerus::misc::UnitTest::tests) {
			std::cout << "# " <<  p.first << std::endl;
		}
		return 0;
	}
	
	
	unsigned passCount=0;
	unsigned totalPassCount=0; 
	unsigned totalCount=0;
	
	for (int currArg = 1; currArg < argc; ++currArg) {
		std::string grp = argv[currArg];
		// do all unit tests
		if (grp == "all") {
			for (const auto &p : *xerus::misc::UnitTest::tests) {
				xerus::misc::internal::print_group_name(p.first);
				passCount=0;
				for (const auto &t : p.second) {
					totalCount += 1;
					if (xerus::misc::internal::test(t)) {
						passCount += 1;
						totalPassCount += 1;
					}
				}
				xerus::misc::internal::print_group_summary(p.first, passCount, p.second.size());
			}
			break;
		}
		
		std::vector<std::string> cmd = xerus::misc::explode(grp,':');
		if (cmd.size()>1) {
			// explicit test inside a group?
			if (cmd.size()>3) {
				std::cout << "########## \033[1;31munknown syntax '" << grp << "'\033[0m" << std::endl;
				continue;
			}
			if (!xerus::misc::UnitTest::tests->count(cmd[0]) || (*xerus::misc::UnitTest::tests)[cmd[0]].count(cmd[1]) == 0) {
				std::cout << "########## \033[1;31munknown unittest '" << cmd[0] << ":" << cmd[1] << "'\033[0m" << std::endl;
				continue;
			}
			xerus::uint64 seed = 0;
			if (cmd.size()>2) {
				// explicit seed given
				seed = std::stoul(cmd[2], nullptr, 16);
			}
			totalCount += 1;
			if (xerus::misc::internal::test({grp, (*xerus::misc::UnitTest::tests)[cmd[0]][cmd[1]]}, seed) ) {
				totalPassCount += 1;
			}
		} else {
			// unknown group
			if (xerus::misc::UnitTest::tests->count(grp) == 0) {
				std::cout << "########## \033[1;31munknown group or unittest '" << grp << "'\033[0m" << std::endl;
				continue;
			}
			// one (whole) group
			xerus::misc::internal::print_group_name(grp);
			passCount=0; 
			for (const auto &t : (*xerus::misc::UnitTest::tests)[grp]) {
				totalCount += 1;
				if (xerus::misc::internal::test(t)) {
					passCount += 1;
					totalPassCount += 1;
				}
			}
			xerus::misc::internal::print_group_summary(grp, passCount, (*xerus::misc::UnitTest::tests)[grp].size());
		}
	}
	
	//Calc total elapsed time
	std::chrono::microseconds::rep totalTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startTime).count();
	
	xerus::misc::internal::print_group_summary("total", totalPassCount, totalCount);
	
	std::cout << "|" << std::endl;
	std::cout << "|" << std::string(23, ' ') << "Total time elapsed: " << static_cast<double>(totalTime)/1000.0 << " ms" << std::string(50, ' ')  << ' ' << std::endl;
	std::cout << "-------------------------------------------------------------------------------" << std::endl;
	
	#ifdef XERUS_TEST_COVERAGE
		xerus::misc::CodeCoverage::print_code_coverage();
	#endif
	
	// Destroy all stored tests to make memory-leak detection simpler
	delete xerus::misc::UnitTest::tests;
	
	return totalPassCount != totalCount;
}
