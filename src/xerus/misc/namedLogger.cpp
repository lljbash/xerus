// Xerus - A General Purpose Tensor Library
// Copyright (C) 2014-2017 Benjamin Huber and Sebastian Wolf. 
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
 * @brief Implementation of the log-buffer functionality and declaration of global logging objects.
 */

#include <time.h>
#include <iomanip>
#include <chrono>
#include <fstream>

#ifdef XERUS_LOG_BUFFER
    #include <fstream>
#endif

#include <xerus/misc/namedLogger.h>
#include <xerus/misc/stringUtilities.h>

namespace xerus { namespace misc { namespace internal {
	std::mutex namedLoggerMutex;
	std::string logFilePrefix;
	bool silenced = false;
	std::chrono::system_clock::time_point programStartTime;
	static void __attribute__((constructor)) initTime() {
		programStartTime = std::chrono::system_clock::now();
	}
	
	void log_timestamp(std::ostream &_out) {
		#ifdef XERUS_LOG_ABSOLUTE_TIME
			//NOTE must not use std::put_time as it was not defined before GCC 5.0
			std::time_t xerus_err_t=std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
			std::tm *xerus_err_ltm=std::localtime(&xerus_err_t);
			_out << std::right << (1900+xerus_err_ltm->tm_year)
				<< '-' << std::setw(2) << std::setfill('0') << (xerus_err_ltm->tm_mon +1)
				<< '-' << std::setw(2) << std::setfill('0') <<  xerus_err_ltm->tm_mday
				<< ' ' << std::setw(2) << std::setfill('0') << xerus_err_ltm->tm_hour
				<< ':' << std::setw(2) << std::setfill('0') <<  xerus_err_ltm->tm_min
				<< ':' << std::setw(2) << std::setfill('0') <<  xerus_err_ltm->tm_sec << ' ' << std::left;
		#else
			static std::chrono::system_clock::time_point startTime = std::max(programStartTime, std::chrono::system_clock::now());
			std::chrono::system_clock::time_point xerus_err_t = std::chrono::system_clock::now();
			auto xerus_err_timediff = std::chrono::duration_cast<std::chrono::milliseconds>(xerus_err_t - startTime).count();
			_out << std::right << '+' << std::setw(2) << std::setfill('0') << (xerus_err_timediff/3600000)
				<< ':' << std::setw(2) << std::setfill('0') << ((xerus_err_timediff/60000)%60)
				<< ':' << std::setw(2) << std::setfill('0') <<  ((xerus_err_timediff/1000)%60)
				<< ',' << std::setw(3) << std::setfill('0') <<  (xerus_err_timediff%1000) << ' ' << std::left;
		#endif
	}
	
	void log_timestamp(std::ostream &_out, const char* _file, int _line, const char* _lvl) {
		log_timestamp(_out);
		_out << std::setfill(' ') << std::setw(20) << std::left << xerus::misc::explode(_file, '/').back() << ':' \
				<< std::right << std::setfill(' ') << std::setw(4) <<_line << " : " \
				<< std::setfill(' ') << std::setw(12) << std::left \
				<< std::string(_lvl) << ": ";
	}
	
	void log_timestamp(std::ostream &_out, const char* _lvl) {
		log_timestamp(_out);
		_out << std::setfill(' ') << std::setw(12) << std::left \
				<< std::string(_lvl) << ": ";
	}
	
	std::ostream &get_fileStream() {
		static std::ofstream fileStream;
		if (!fileStream || !fileStream.is_open()) {
			fileStream.close();
			fileStream.open("error.log", std::ofstream::app | std::ofstream::out);
		}
		return fileStream;
	}
	
	namespace buffer {
		std::stringstream current;
		std::stringstream old;
		
		void checkSwitch() {
			if (current.str().size() > 1024*1024) {
				#if defined(__GNUC__) && __GNUC__ < 5
					old.str(current.str());
					current.str(std::string());
					current.clear();
				#else
					old = std::move(current);
					current = std::stringstream();
				#endif
			}
		}
		
		void dump_log(std::string _comment)  {
			std::string name = std::string("errors/") + logFilePrefix + std::to_string(std::time(nullptr)) + ".txt";
			std::ofstream out(name, std::ofstream::out);
			out << "Error: " << _comment << std::endl << std::endl;

			// Get callstack
			out << "-------------------------------------------------------------------------------" << std::endl 
				<< "  Callstack : " << std::endl
				<< "-------------------------------------------------------------------------------" << std::endl
				<< get_call_stack() << std::endl;
			
			// Output namedLogger
			old.flush();
			current.flush();
			out << "-------------------------------------------------------------------------------" << std::endl 
				<< "  last " << (current.str().size() + old.str().size()) << " bytes of log:" << std::endl
				<< "-------------------------------------------------------------------------------" << std::endl 
				<< old.str() << current.str() << "horst" << std::endl; 
			out.close();
		}
	} // namespace buffer
} // namespace internal
} // namespace misc
} // namespace xerus
