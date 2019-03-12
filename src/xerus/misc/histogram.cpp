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
 * @brief Implementation of the histogram classes.
 */

#include <xerus/misc/histogram.h>
#include <xerus/misc.h>
#include <xerus/misc/internal.h>

namespace xerus { namespace misc {

	LogHistogram::LogHistogram(const double _base) : base(_base), totalCount(0) {}
	
	LogHistogram &LogHistogram::operator+=(const LogHistogram &_other) {
		REQUIRE(misc::approx_equal(_other.base, base), "only histograms of identical base can be added");
		for (const auto &h : _other.buckets) {
			buckets[h.first] += h.second;
		}
		totalCount += _other.totalCount;
		return *this;
	}
	
	void LogHistogram::add(double _value, size_t _count) {
		double logRate = log(_value)/log(base);
		REQUIRE(std::isfinite(_value), "tried to add illegal value " << _value << " into a histogram");
		buckets[int(logRate)] += _count;
		totalCount += _count;
	}
	
	LogHistogram LogHistogram::read_from_file(const std::string &_fileName) {
		std::ifstream in(_fileName);
		LogHistogram result(0.0);
		std::string line;
		std::getline(in, line);
		REQUIRE(line == "# raw data:", "unknown histogram file format " << _fileName); //TODO this should throw an exception
		char c;
		in >> c;
		REQUIRE(c == '#', "missing information in histogram file " << _fileName);
		in >> result.base >> result.totalCount;
		std::getline(in, line); // read in rest of this line
		std::getline(in, line); // line now contains all buckets
		std::stringstream bucketData(line);
		bucketData >> c;
		REQUIRE(c == '#', "missing information in histogram file " << _fileName);
		int bucketIndex;
		while (bucketData >> bucketIndex) {
			size_t count;
			if (!(bucketData >> count)) {
				LOG(fatal, "missing bucket count in histogram file " << _fileName);
			}
			result.buckets[bucketIndex] = count;
		}
		size_t accountedTime=0;
		for (auto &h : result.buckets) {
			accountedTime += h.second;
		}
		REQUIRE(accountedTime == result.totalCount, "histogram data inconsistent in file " << _fileName);
		return result;
	}
	
	void LogHistogram::dump_to_file(const std::string &_fileName) const {
		std::ofstream out(_fileName);
		out << "# raw data:\n";
		out << "# " << base << ' ' << totalCount << '\n';
		out << '#';
		for (auto &h : buckets) {
			out << ' ' << h.first << ' ' << h.second;
		}
		out << "\n# plotable data:\n";
		if (!buckets.empty()) {
			int firstOutput = buckets.begin()->first - 1;
			int lastOutput = buckets.rbegin()->first + 1;
			for (int i=firstOutput; i<=lastOutput; ++i) {
				out << pow(base, i) << ' ';
				if (buckets.count(i) > 0) {
					out << double(buckets.find(i)->second)/double(totalCount) << '\n';
				} else {
					out << "0\n";
				}
			}
		}
		out.close();
	}

} // namespace misc
} // namespace xerus

