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
 * @brief Implementation of performanceAnalysis::get_analysis.
 */

#include <xerus/misc/performanceAnalysis.h>
#include <sstream>
#include <iomanip>

namespace xerus {
	namespace misc {
		namespace performanceAnalysis {
			#ifdef XERUS_PERFORMANCE_ANALYSIS
				const size_t startupTime = uTime();
				std::map<std::string, std::map<std::string, std::map<std::string, std::pair<size_t, size_t>>>> calls;
				
				std::string get_analysis() {
					const size_t totalTime = uTime() - startupTime;
					size_t totalExplainedTime = 0;
					
					std::stringstream mainStream;
					mainStream << std::endl;
					mainStream << "| ==================================================================================" << std::endl;
					mainStream << "| ============================== Performance Analysis ==============================" << std::endl;
					mainStream << "| ==================================================================================" << std::endl << std::endl;

					for(const auto& group : calls) {
						size_t totalGroupCalls = 0, totalGroupTime = 0;
						std::stringstream groupStream;
						
						for(const auto& call : group.second) {
							size_t totalCallCalls = 0, totalCallTime = 0;
							std::stringstream callStream;
							
							for(const auto& subCall : call.second) {
								totalGroupCalls += subCall.second.first; totalCallCalls += subCall.second.first;
								totalGroupTime += subCall.second.second; totalCallTime += subCall.second.second;
								if(1000*subCall.second.second > totalTime) {
									callStream << "| | " << std::setfill (' ') << std::setw(10) << subCall.second.second/1000
									<< " ms ( " << std::setw(3) << (100*subCall.second.second)/totalTime << "% ) in " << std::setfill (' ') << std::setw(10) << subCall.second.first 
									<< " calls (" << std::setfill (' ') << std::setw(8) << subCall.second.second/(1000*subCall.second.first)
									<< " ms in average) for " << subCall.first << std::endl;
								}
							}
							if(1000*totalCallTime > totalTime) {
								groupStream << std::endl;
								groupStream << "| --------------------------- " << std::left << std::setfill (' ') << std::setw(26) << call.first << " ---------------------------" << std::endl;
								groupStream << callStream.str();
								groupStream << "| Together " << std::setw(10) << std::right << totalCallTime/1000 << " ms (" << std::setw(3) << (100*totalCallTime)/totalTime << "% ) in " << std::setw(8) << totalCallCalls << " calls." << std::endl;
							}
						}
						if(1000*totalGroupTime > totalTime) {
							mainStream << std::endl;
							mainStream << "| ============================== " << std::left << std::setfill (' ') << std::setw(20) << group.first << " ==============================" << std::endl;
							mainStream << "| ============ Total time " << std::setw(14) << std::right << totalGroupTime/1000 << " ms (" << std::setw(3) << (100*totalGroupTime)/totalTime << "% ) in " << std::setw(10) << totalGroupCalls << " calls ============" << std::endl;
							mainStream << groupStream.str();
						}
						totalExplainedTime += totalGroupTime;
					}
					
					mainStream << std::endl << "| The analysed low level functions explain " << (100*totalExplainedTime)/(totalTime) << "% of the total time." << std::endl << std::endl;
					return mainStream.str();
				}
			#else 
				std::string get_analysis() { return "XERUS_PERFORMANCE_ANALYSIS must be set to obtain an analysis."; }
			#endif
		} // namespace performanceAnalysis
	} // namespace misc
} // namespace xerus
