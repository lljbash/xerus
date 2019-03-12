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
 * @brief Implementation of the non-template basic string functionality defined in stringUtilities.h.
 */

#include <xerus/misc/stringUtilities.h>
#include <memory>
#include <sstream>
#include <cxxabi.h>

namespace xerus {
	namespace misc {
		
		std::string demangle_cxa(const std::string &_cxa) {
			int status;
			std::unique_ptr<char, void(*)(void*)> realname(abi::__cxa_demangle(_cxa.data(), nullptr, nullptr, &status), &free);
			if (status != 0) { return _cxa; }

			if (realname) { 
				return std::string(realname.get()); 
			} 
			return ""; 
		}

		std::string normalize_pathname(const std::string &_name) {
			std::vector<std::string> oldpath = explode(_name,'/');
			std::vector<std::string *> newpath;
			for (std::string &f : oldpath) {
				if (f.empty()) { continue; }
				if (f==".." && !newpath.empty() && *newpath.back() != "..") {
					newpath.pop_back();
				} else {
					newpath.push_back(&f);
				}
			}
			std::string ret;
			for (std::string *f : newpath) {
				ret += *f;
				ret += '/';
			}
			if (!ret.empty()) { ret.pop_back(); }
			return ret;
		}

		std::vector<std::string> explode(const std::string& _string, const char _delim) {
			std::vector<std::string> result;
			std::istringstream iss(_string);

			std::string token;
			while(std::getline(iss, token, _delim)) {
				result.push_back(std::move(token));
			}

			return result;
		}

		void replace(std::string& _string, const std::string& _search, const std::string& _replace) {
			size_t pos = 0;
			while((pos = _string.find(_search, pos)) != std::string::npos){
				_string.replace(pos, _search.length(), _replace);
				pos += _replace.length();
			}
		}
		
		std::string trim(const std::string& _string, const std::string& whitespace) {
			const size_t strBegin = _string.find_first_not_of(whitespace);
			if (strBegin == std::string::npos) {
				return "";
			} 
			const size_t strEnd = _string.find_last_not_of(whitespace);
			return _string.substr(strBegin, strEnd - strBegin + 1);
		}

		std::string reduce(const std::string& _string, const std::string& whitespace, const std::string& fill) {
			// Trim first
			auto trimedString = trim(_string, whitespace); 

			// Replace sub ranges
			auto beginSpace = trimedString.find_first_of(whitespace);
			while (beginSpace != std::string::npos) {
				const auto endSpace = trimedString.find_first_not_of(whitespace, beginSpace);
				const auto range = endSpace - beginSpace;

				trimedString.replace(beginSpace, range, fill);

				const auto newStart = beginSpace + fill.length();
				beginSpace = trimedString.find_first_of(whitespace, newStart);
			}
			return trimedString;
		}
	} // namespace misc
} // namespace xerus
