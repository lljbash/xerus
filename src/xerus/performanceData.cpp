// Xerus - A General Purpose Tensor Library
// Copyright (C) 2014-2018 Benjamin Huber and Sebastian Wolf. 
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
* @brief Implementation of the PerformanceData class.
*/

#include <string>
#include <fstream>
#include <xerus/ttNetwork.h>
#include <xerus/performanceData.h>
#include <xerus/misc/internal.h>
 

namespace xerus {
	
	PerformanceData::PerformanceData(const bool _printProgress, const bool _active) : 
		active(_active), printProgress(_printProgress), startTime(~0ul), stopTime(~0ul) {}
		
		
	PerformanceData::PerformanceData(const ErrorFunction& _errorFunction, const bool _printProgress, const bool _active) : 
		active(_active), printProgress(_printProgress), errorFunction(_errorFunction), startTime(~0ul), stopTime(~0ul) {}
	
	
	void PerformanceData::start() {
		if (active) {
			if(printProgress) {
				std::stringstream ss(additionalInformation);
				while (ss) {
					std::string line;
					std::getline(ss, line);
					XERUS_LOG_SHORT(PerformanceData, line);
				}
			}
			startTime = misc::uTime();
		}
	}
	
	
	void PerformanceData::stop_timer() {
		if (active) {
			stopTime = misc::uTime();
		}
	}
	
	
	void PerformanceData::continue_timer() {
		if (active) {
			size_t currtime = misc::uTime();
			startTime += currtime - stopTime;
			stopTime = ~0ul;
		}
	}
	
	
	void PerformanceData::reset() {
		if (active) {
			data.clear();
			additionalInformation.clear();
			startTime = ~0ul;
			stopTime = ~0ul;
		}
	}
	
	
	size_t PerformanceData::get_elapsed_time() const {
		return misc::uTime() - startTime;
	}
	
	
	size_t PerformanceData::get_runtime() const {
		if (stopTime != ~0ul) {
			return stopTime - startTime;
		} else {
			return misc::uTime() - startTime;
		}
	}
	
	
	void PerformanceData::add(const value_t _residual, const TTTensor& _x, const size_t _flags) {
		if (data.empty()) {
			add(0, std::vector<double>(1, _residual), _x, _flags);
		} else {
			add(data.back().iteration+1, std::vector<double>(1, _residual), _x, _flags);
		}
	}
	
	
	void PerformanceData::add(const std::vector<double>& _residuals, const TTTensor& _x, const size_t _flags) {
		if (data.empty()) {
			add(0, _residuals, _x, _flags);
		} else {
			add(data.back().iteration+1, _residuals, _x, _flags);
		}
	}
	
	
	void PerformanceData::add(const size_t _itrCount, double _residual, const TTTensor& _x, const size_t _flags) {
		add(_itrCount, std::vector<double>(1, _residual), _x, _flags);
	}
	
	
	void PerformanceData::add(const size_t _itrCount, const std::vector<double>& _residuals, const TTTensor& _x, const size_t _flags) {
		if (active) {
			if (startTime == ~0ul) {
				start();
			}
			stop_timer();
			
			REQUIRE(!_residuals.empty(), "Need at least one residual");
			
			const double error = errorFunction ? errorFunction(_x) : 0.0;
			data.emplace_back(_itrCount, get_elapsed_time(), _residuals, error, _x.degrees_of_freedom(), _flags);

			
			if (printProgress) {
				if(errorFunction) {
					LOG_SHORT(PerformanceData, "Iteration " << std::setw(4) << std::setfill(' ') << _itrCount 
						<< " Time: " << std::right << std::setw(6) << std::setfill(' ') << std::fixed << std::setprecision(2) << double(data.back().elapsedTime)*1e-6
						<< "s Residuals: " <<  std::setw(11) << std::setfill(' ') << std::scientific << std::setprecision(6) << data.back().residuals
						<< " Error: " << std::setw(11) << std::setfill(' ') << std::scientific << std::setprecision(6) << data.back().error
						<< " Dofs: " << data.back().dofs << " Flags: " << _flags);
				} else {
					LOG_SHORT(PerformanceData, "Iteration " << std::setw(4) << std::setfill(' ') << _itrCount 
						<< " Time: " << std::right << std::setw(6) << std::setfill(' ') << std::fixed << std::setprecision(2) << double(data.back().elapsedTime)*1e-6
						<< "s Residuals: " <<  std::setw(11) << std::setfill(' ') << std::scientific << std::setprecision(6) << data.back().residuals
						<< " Dofs: " << data.back().dofs << " Flags: " << _flags);
				}
			}
			continue_timer();
		}
	}
	
	
	void PerformanceData::dump_to_file(const std::string &_fileName) const {
		REQUIRE(active && !data.empty(), "Inactive or empty PerformanceData cannot be dumped to file");
		std::string header;
		header += "# ";
		header += additionalInformation;
		misc::replace(header, "\n", "\n# ");
		std::ofstream out(_fileName);
		out << header;
		out << "\n#itr \ttime[us] \tresiduals("<<data.back().residuals.size()<<") \terror \tdofs \tflags \n";
		for (const auto& d : data) {
			out << d.iteration << '\t' << d.elapsedTime << '\t';
			for (const auto r : d.residuals) {
				out << r << '\t';
			}
			
			out << d.error << '\t' << d.dofs << '\t' << d.flags;
			out << '\n';
		}
		out.close();
	}
	
	misc::LogHistogram PerformanceData::get_histogram(const xerus::value_t _base, bool _assumeConvergence) const {
		misc::LogHistogram hist(_base);
		std::vector<PerformanceData::DataPoint> convergenceData(data);
		if (_assumeConvergence) {
			value_t finalResidual = data.back().residuals[0];
			convergenceData.pop_back();
			for (auto &p : convergenceData) {
				p.residuals[0] -= finalResidual;
			}
		}
		
		for (size_t i = 1; i<convergenceData.size(); ++i) {
			if (convergenceData[i].residuals[0] <= 0 || convergenceData[i-1].residuals[0] <= 0
				|| convergenceData[i].residuals[0] >= convergenceData[i-1].residuals[0]) {
				continue;
			}
			
			// assume x_2 = x_1 * 2^(-alpha * delta-t)
			value_t relativeChange = convergenceData[i].residuals[0]/convergenceData[i-1].residuals[0];
			value_t exponent = log(relativeChange) / log(2);
			size_t delta_t = convergenceData[i].elapsedTime - convergenceData[i-1].elapsedTime;
			if (delta_t == 0) {
				LOG(warning, "approximated 0us by 1us");
				delta_t = 1;
			}
			value_t rate = - exponent / value_t(delta_t);
			REQUIRE(std::isfinite(rate), "infinite rate? " << relativeChange << " " << exponent << " " << delta_t << " " << rate);
			hist.add(rate, delta_t);
		}
		return hist;
	}

	PerformanceData NoPerfData(false, false);
} // namespace xerus
