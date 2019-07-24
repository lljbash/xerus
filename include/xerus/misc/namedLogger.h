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
 * @brief Header file for all logging macros and log-buffer functionality.
 */

#pragma once

#include <mutex>
#include <string>
#include <sstream>

#include "callStack.h"
#include "exceptions.h"

#ifdef XERUS_LOGFILE
    #include <fstream>
    #define XERUS_LOGSTREAM xerus::misc::internal::get_fileStream()
#else
    #include <iostream>
    #define XERUS_LOGSTREAM std::cerr
#endif

namespace xerus {
    namespace misc {
        namespace internal {
			/**
			 * @brief Hashes a given c-string using the FNV-1a standard hash.
			 * @details This is used eg. to use strings as template arguments.
			 */
			constexpr uint64_t log_namehash(const char* x) {
				return *x ? (uint64_t(*x) ^ xerus::misc::internal::log_namehash(x+1))*0x100000001B3ul : 0xCBF29CE484222325ul;
			}
			
			void log_timestamp(std::ostream &_out, const char* _file, int _line, const char* _lvl);
			void log_timestamp(std::ostream &_out, const char* _lvl);
			void log_timestamp(std::ostream &_out);
			std::ostream &get_fileStream();
			
			// If the LOG_BUFFER is active there is the additional option only to print the log if an error occurs.
			#ifdef XERUS_LOG_BUFFER
				enum LogLvl {
					NOT_LOGGING = 0,
					LOGGING_ON_ERROR = 1,
					LOGGING_FULL = 2,
					LOGGING_EXCEPTION = 3
				};
			#else
				enum LogLvl {
					NOT_LOGGING = 0,
					LOGGING_FULL = 2,
					LOGGING_EXCEPTION = 3
				};
				static const auto LOGGING_ON_ERROR = NOT_LOGGING;
			#endif

            extern std::mutex namedLoggerMutex;
            extern std::string logFilePrefix;
            extern bool silenced;
			extern std::chrono::system_clock::time_point programStartTime;
			
			namespace buffer {
				extern std::stringstream current;
				extern std::stringstream old;
				
				void checkSwitch();
				
				void dump_log(std::string _comment);
			}
        }
    }
}

#define XERUS_STRINGIFY2( x) #x
#define XERUS_STRINGIFY(x) XERUS_STRINGIFY2(x)


/**
 * @def XERUS_COMPILE_TIME_EVAL(e)
 * @brief forces the compiler to evaluate @a e during compilation
 */
#define XERUS_COMPILE_TIME_EVAL(e) (std::integral_constant<decltype(e), e>::value)


// in the following note that:
// - everything is piped into a stringstream for two reasons: 1. so that functions appearing in msg will only be executed once
//                                                            2. so that buffer(pointers) can be piped as usual even though they empty the buffer while being piped
// - in performance critical programs logging should be disabled completely, the mutex in the following code will block the whole function. a read/write mutex could
//   increase performance a bit (by only making sure that the delete inside checkSwitch is not called while another thread is piping something) and the mutex
//   in the LOG_NO_BUFFER version could be disabled completely (the streams are thread"safe" (only the output will be jumbled a bit)

#ifdef XERUS_LOG_BUFFER
    #define XERUS_NAMED_LOGGER_LOGBUFFER(lvl) \
		if (::xerus::misc::internal::LogFlag<xerus::misc::internal::log_namehash(XERUS_STRINGIFY(lvl))>::flag != xerus::misc::internal::NOT_LOGGING && !xerus::misc::internal::silenced) { \
			::xerus::misc::internal::log_timestamp(xerus::misc::internal::buffer::current, __FILE__, __LINE__, XERUS_STRINGIFY(lvl)); \
			xerus::misc::internal::buffer::current << tmpStream.str(); \
			xerus::misc::internal::buffer::checkSwitch(); \
			if (XERUS_COMPILE_TIME_EVAL(xerus::misc::internal::log_namehash(XERUS_STRINGIFY(lvl))==xerus::misc::internal::log_namehash("error"))) { \
				xerus::misc::internal::buffer::dump_log(std::string("error invoked:\n")+tmpStream.str()); \
			} \
			if (XERUS_COMPILE_TIME_EVAL(xerus::misc::internal::log_namehash(XERUS_STRINGIFY(lvl))==xerus::misc::internal::log_namehash("critical")))  { \
				xerus::misc::internal::buffer::dump_log(std::string("critical error invoked:\n")+tmpStream.str()); \
			} \
			if (XERUS_COMPILE_TIME_EVAL(xerus::misc::internal::log_namehash(XERUS_STRINGIFY(lvl))==xerus::misc::internal::log_namehash("fatal"))) { \
				xerus::misc::internal::buffer::dump_log(std::string("fatal error invoked:\n")+tmpStream.str()); \
			} \
		}
#else // no log buffer
    #define XERUS_NAMED_LOGGER_LOGBUFFER(lvl)
#endif
		

		
		
/**
 * @def XERUS_LOG(lvl, msg)
 * @brief logs the message @a msg with severity level @a lvl
 * @details the exact behaviour is modified by the SET_DEFAULT_LOG_LEVELS and XERUS_SET_LOGGING macros. In case @a lvl is not being logged with the
 *   current configuration, this macro evaluates to an `if (false) {}` expression and is fully removed by the compiler.
 */
#define XERUS_LOG(lvl, ...) \
    if (::xerus::misc::internal::LogFlag<xerus::misc::internal::log_namehash(XERUS_STRINGIFY(lvl))>::flag != xerus::misc::internal::NOT_LOGGING) { \
        std::stringstream tmpStream; \
        tmpStream << __VA_ARGS__ << std::endl; \
        xerus::misc::internal::namedLoggerMutex.lock(); \
        if (::xerus::misc::internal::LogFlag<xerus::misc::internal::log_namehash(XERUS_STRINGIFY(lvl))>::flag == xerus::misc::internal::LOGGING_FULL && !xerus::misc::internal::silenced) { \
            ::xerus::misc::internal::log_timestamp(XERUS_LOGSTREAM, __FILE__, __LINE__, XERUS_STRINGIFY(lvl)); \
			XERUS_LOGSTREAM << tmpStream.str() << std::flush; \
        } \
        XERUS_NAMED_LOGGER_LOGBUFFER(lvl) \
        xerus::misc::internal::namedLoggerMutex.unlock(); \
        if (::xerus::misc::internal::LogFlag<xerus::misc::internal::log_namehash(XERUS_STRINGIFY(lvl))>::flag == xerus::misc::internal::LOGGING_EXCEPTION ) { \
			XERUS_THROW(xerus::misc::generic_error() << __FILE__ << ":" << __LINE__ << ": " XERUS_STRINGIFY(lvl) " invoked:\n" << tmpStream.str() << "callstack:\n" << xerus::misc::get_call_stack()); \
        } \
    } else \
        (void)0

/**
 * @def XERUS_LOG_SHORT(lvl, msg)
 * @brief logs the message @a msg with severity level @a lvl, omits the current file name and line number
 * @details the exact behaviour is modified by the SET_DEFAULT_LOG_LEVELS and XERUS_SET_LOGGING macros. In case @a lvl is not being logged with the
 *   current configuration, this macro evaluates to an `if (false) {}` expression and is fully removed by the compiler.
 */
#define XERUS_LOG_SHORT(lvl, ...) \
    if (::xerus::misc::internal::LogFlag<xerus::misc::internal::log_namehash(XERUS_STRINGIFY(lvl))>::flag != xerus::misc::internal::NOT_LOGGING) { \
        std::stringstream tmpStream; \
        tmpStream << __VA_ARGS__ << std::endl; \
        xerus::misc::internal::namedLoggerMutex.lock(); \
        if (::xerus::misc::internal::LogFlag<xerus::misc::internal::log_namehash(XERUS_STRINGIFY(lvl))>::flag == xerus::misc::internal::LOGGING_FULL && !xerus::misc::internal::silenced) { \
            ::xerus::misc::internal::log_timestamp(XERUS_LOGSTREAM, XERUS_STRINGIFY(lvl)); \
			XERUS_LOGSTREAM << tmpStream.str() << std::flush; \
        } \
        XERUS_NAMED_LOGGER_LOGBUFFER(lvl) \
        xerus::misc::internal::namedLoggerMutex.unlock(); \
        if (::xerus::misc::internal::LogFlag<xerus::misc::internal::log_namehash(XERUS_STRINGIFY(lvl))>::flag == xerus::misc::internal::LOGGING_EXCEPTION ) { \
            XERUS_THROW(xerus::misc::generic_error() << XERUS_STRINGIFY(lvl) " error invoked:\n" << tmpStream.str() << "callstack:\n" << xerus::misc::get_call_stack()); \
        } \
    } else \
        (void)0

/**
 * @def XERUS_LOG_ONCE(lvl, msg)
 * @brief logs the message @a msg with severity level @a lvl at most once per program execution
 * @details the exact behaviour is modified by the SET_DEFAULT_LOG_LEVELS and XERUS_SET_LOGGING macros. In case @a lvl is not being logged with the
 *   current configuration, this macro evaluates to an `if (false) {}` expression and is fully removed by the compiler.
 */
#define XERUS_LOG_ONCE(lvl, ...) \
	{\
		static bool logged = false;\
		if (!logged) {\
			XERUS_LOG(lvl, __VA_ARGS__);\
			logged = true;\
		}\
	}

/**
 * @def XERUS_IS_LOGGING(lvl)
 * @brief evaluates to true if @a lvl is begin logged (either to cerr or into a file on error) in the current configuration 
 */
#define XERUS_IS_LOGGING(lvl) \
    (::xerus::misc::internal::LogFlag<xerus::misc::internal::log_namehash(XERUS_STRINGIFY(lvl))>::flag != xerus::misc::internal::NOT_LOGGING)


namespace xerus { namespace misc { namespace internal {
	template<uint64_t lvl> struct LogFlag { static const int flag = xerus::misc::internal::LOGGING_FULL; };
}}}

/**
 * @def XERUS_SET_LOGGING(lvl, value)
 * @brief set the logging behaviour of severity level @a lvl to @a value (either NOT_LOGGING, LOGGING_ON_ERROR or LOGGING_FULL)
 * @details this definition must not be repeated and must be defined in a global header that is included before any msg is logged with that lvl
 */
#define XERUS_SET_LOGGING(lvl, value) \
	namespace xerus { namespace misc { namespace internal { \
		template<> struct LogFlag<xerus::misc::internal::log_namehash(XERUS_STRINGIFY(lvl))>{ static const int flag = value; }; \
    }}}



// Default log levels  
#define XERUS_SET_DEFAULT_LOG_LEVELS \
	XERUS_SET_LOGGING(fatal, 		xerus::misc::internal::LOGGING_EXCEPTION)\
	XERUS_SET_LOGGING(critical, 	xerus::misc::internal::LOGGING_EXCEPTION)\
	XERUS_SET_LOGGING(error, 		xerus::misc::internal::LOGGING_EXCEPTION)\
	XERUS_SET_LOGGING(warning, 		xerus::misc::internal::LOGGING_FULL)\
	XERUS_SET_LOGGING(info, 		xerus::misc::internal::LOGGING_FULL)\
	XERUS_SET_LOGGING(debug, 		xerus::misc::internal::LOGGING_ON_ERROR)

#ifdef XERUS_LOG_ERROR
#undef XERUS_SET_DEFAULT_LOG_LEVELS
#define XERUS_SET_DEFAULT_LOG_LEVELS \
	XERUS_SET_LOGGING(fatal, 		xerus::misc::internal::LOGGING_EXCEPTION)\
	XERUS_SET_LOGGING(critical, 	xerus::misc::internal::LOGGING_EXCEPTION)\
	XERUS_SET_LOGGING(error, 		xerus::misc::internal::LOGGING_EXCEPTION)\
	XERUS_SET_LOGGING(warning, 		xerus::misc::internal::LOGGING_ON_ERROR)\
	XERUS_SET_LOGGING(info, 		xerus::misc::internal::LOGGING_ON_ERROR)\
	XERUS_SET_LOGGING(debug, 		xerus::misc::internal::LOGGING_ON_ERROR)
#endif
    
#ifdef XERUS_LOG_WARNING
#undef XERUS_SET_DEFAULT_LOG_LEVELS
#define XERUS_SET_DEFAULT_LOG_LEVELS \
	XERUS_SET_LOGGING(fatal, 		xerus::misc::internal::LOGGING_EXCEPTION)\
	XERUS_SET_LOGGING(critical, 	xerus::misc::internal::LOGGING_EXCEPTION)\
	XERUS_SET_LOGGING(error, 		xerus::misc::internal::LOGGING_EXCEPTION)\
	XERUS_SET_LOGGING(warning, 		xerus::misc::internal::LOGGING_FULL)\
	XERUS_SET_LOGGING(info, 		xerus::misc::internal::LOGGING_ON_ERROR)\
	XERUS_SET_LOGGING(debug, 		xerus::misc::internal::LOGGING_ON_ERROR)
#endif
    
#ifdef XERUS_LOG_INFO
#undef XERUS_SET_DEFAULT_LOG_LEVELS
#define XERUS_SET_DEFAULT_LOG_LEVELS \
	XERUS_SET_LOGGING(fatal, 		xerus::misc::internal::LOGGING_EXCEPTION)\
	XERUS_SET_LOGGING(critical, 	xerus::misc::internal::LOGGING_EXCEPTION)\
	XERUS_SET_LOGGING(error, 		xerus::misc::internal::LOGGING_EXCEPTION)\
	XERUS_SET_LOGGING(warning, 		xerus::misc::internal::LOGGING_FULL)\
	XERUS_SET_LOGGING(info, 		xerus::misc::internal::LOGGING_FULL)\
	XERUS_SET_LOGGING(debug, 		xerus::misc::internal::LOGGING_ON_ERROR)
#endif
    
#ifdef XERUS_LOG_DEBUG
#undef XERUS_SET_DEFAULT_LOG_LEVELS
#define XERUS_SET_DEFAULT_LOG_LEVELS \
	XERUS_SET_LOGGING(fatal, 		xerus::misc::internal::LOGGING_EXCEPTION)\
	XERUS_SET_LOGGING(critical, 	xerus::misc::internal::LOGGING_EXCEPTION)\
	XERUS_SET_LOGGING(error, 		xerus::misc::internal::LOGGING_EXCEPTION)\
	XERUS_SET_LOGGING(warning, 		xerus::misc::internal::LOGGING_FULL)\
	XERUS_SET_LOGGING(info, 		xerus::misc::internal::LOGGING_FULL)\
	XERUS_SET_LOGGING(debug, 		xerus::misc::internal::LOGGING_FULL)
#endif

XERUS_SET_DEFAULT_LOG_LEVELS
