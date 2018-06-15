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
* @brief Header file for the standard contaienr support functions.
*/

#pragma once

#include <vector>
#include <set>
#include <map>
#include <tuple>
#include <algorithm>
#include <numeric>
#include <iostream>

#include "standard.h"
#include "sfinae.h"
#include "check.h"

namespace xerus { namespace misc {
	
	#if __GNUC__ > 4 || defined(__clang__)
		
		XERUS_GENERATE_HAS_FUNCTION(count)
		XERUS_GENERATE_HAS_FUNCTION(find)
	
		
		///@brief: Counts how often an element is contained in an arbitary container
		template<template<class, class...> class container_t, class item_t, class... rest_t,
			typename std::enable_if<sfinae::has_count<container_t<item_t, rest_t...>, item_t>::value, int>::type = 0>
		inline size_t count(const container_t<item_t, rest_t...> &_container, const item_t &_item) noexcept(noexcept(_container.count(_item))) { 
			return _container.count(_item);
		}
		
		
		///@brief: Counts how often an element is contained in an arbitary container
		template<template<class, class...> class container_t, class item_t, class... rest_t, 
		typename std::enable_if<!sfinae::has_count<container_t<item_t, rest_t...>, item_t>::value, int>::type = 0>
		inline size_t count(const container_t<item_t, rest_t...> &_container, const item_t &_item) {
			size_t count = 0;
			for(const item_t& otherItem : _container) {
				if(otherItem == _item) { count++; }
			}
			return count;
		}
		
		
		///@brief: Checks whether an arbitary container contains a certain element.
		template<template<class, class...> class container_t, class item_t, class... rest_t,
			typename std::enable_if<sfinae::has_find<container_t<item_t, rest_t...>, item_t>::value, int>::type = 0>
		inline bool contains(const container_t<item_t, rest_t...> &_container, const item_t &_item) {
			return _container.find(_item) != _container.end();
		}

		
		///@brief: Checks whether an arbitary container contains a certain element.
		template<template<class, class...> class container_t, class item_t, class... rest_t,
			typename std::enable_if<!sfinae::has_find<container_t<item_t, rest_t...>, item_t>::value, int>::type = 0>
		inline bool contains(const container_t<item_t, rest_t...> &_container, const item_t &_item) {
			return std::find(_container.begin(), _container.end(), _item) != _container.end();
		}

	#else 
		
		XERUS_GENERATE_HAS_MEMBER(count)
		
		///@brief: Counts how often an element is contained in an arbitary container
		template<template<class, class...> class container_t, class item_t, class... rest_t, 
			typename std::enable_if<!has_member_count<container_t<item_t, rest_t...>>::value, int>::type = 0>
		inline size_t count(const container_t<item_t, rest_t...> &_container, const item_t &_item) {
			size_t count = 0;
			for(const item_t& otherItem : _container) {
				if(otherItem == _item) { count++; }
			}
			return count;
		}
		
		
		///@brief: Counts how often an element is contained in an arbitary container
		template<template<class, class...> class container_t, class item_t, class... rest_t, 
			typename std::enable_if<has_member_count<container_t<item_t, rest_t...>>::value, int>::type = 0>
		inline size_t count(const container_t<item_t, rest_t...> &_container, const item_t &_item) {
			return _container.count(_item);
		}
		
		///@brief: Checks whether an arbitary container contains a certain element.
		template<template<class, class...> class container_t, class item_t, class... rest_t, 
			typename std::enable_if<!has_member_count<container_t<item_t, rest_t...>>::value, int>::type = 0>
		inline bool contains(const container_t<item_t, rest_t...> &_container, const item_t &_item) {
			return std::find(_container.begin(), _container.end(), _item) != _container.end();
		}
		
		
		///@brief: Checks whether an arbitary container contains a certain element.
		template<template<class, class...> class container_t, class item_t, class... rest_t, 
			typename std::enable_if<has_member_count<container_t<item_t, rest_t...>>::value, int>::type = 0>
		inline bool contains(const container_t<item_t, rest_t...> &_container, const item_t &_item) {
			return _container.find(_item) != _container.end();
		}
	
	#endif
	
	///@brief:  Check whether an arbitary container contains all elemets of another arbitary container.
	template<template<class, class...> class containerA_t, template<class, class...> class containerB_t, class item_t, class... restA_t, class... restB_t>
	inline bool contains(const containerA_t<item_t, restA_t...> &_largeContainer, const containerB_t<item_t, restB_t...> &_smallContainer) {
		for(const item_t &item : _smallContainer) {
			if(!contains(_largeContainer, item)) { return false; }
		}
		return true;
	}
	
	
	///@brief: Checks whether two arbitary containers are disjunct, i.e. share no object.
	template<template<class, class...> class container_t, class item_t, class... rest_t>
	inline bool disjunct(const container_t<item_t, rest_t...>& _containerA, const container_t<item_t, rest_t...>& _containerB) {
		for(const item_t& item : _containerA) {
			if(contains(_containerB, item)) { return false; }
		}
		return true;
	}
	
	
	///@brief: Selects the maximal element from an arbitary container.
	template<template<class, class...> class container_t, class item_t, class... rest_t>
	inline item_t max(const container_t<item_t, rest_t...>& _container) {
		XERUS_REQUIRE(!_container.empty(), "max() must not be invoked with empty container.");
		return *std::max_element(_container.begin(), _container.end());
	}
	

	///@brief: Selects the minimal element from an arbitary container.
	template<template<class, class...> class container_t, class item_t, class... rest_t>
	inline item_t min(const container_t<item_t, rest_t...>& _container) {
		XERUS_REQUIRE(!_container.empty(), "min() must not be invoked with empty container.");
		return *std::min_element(_container.begin(), _container.end());
	}
	
	
	///@brief: Calculates the sum of all entries of an arbitary container.
	template<template<class, class...> class container_t, class item_t, class... rest_t>
	inline item_t sum(const container_t<item_t, rest_t...>& _container) {
		return std::accumulate(_container.begin(), _container.end(), item_t(0));
	}
	
	///@brief: Calculates the average of all entries of an arbitary container.
	template<template<class, class...> class container_t, class item_t, class... rest_t>
	inline item_t average(const container_t<item_t, rest_t...>& _container) {
		return sum(_container)/_container.size();
	}
	

	///@brief: Calculates the product of all entries of an arbitary container.
	template<template<class, class...> class container_t, class item_t, class... rest_t>
	inline item_t product(const container_t<item_t, rest_t...>& _container) {
		return std::accumulate(_container.begin(), _container.end(), item_t(1), std::multiplies<item_t>());
	}
	
	
	///@brief: Calculates the floating point product of all entries of an arbitary container.
	template<template<class, class...> class container_t, class item_t, class... rest_t>
	inline double fp_product(const container_t<item_t, rest_t...>& _container) {
		return std::accumulate(_container.begin(), _container.end(), double(1.0), std::multiplies<double>());
	}
	
	
	///@brief: Calculates the product the entries in the range [_first, _last).
	template<class item_t, class... rest_t>
	inline item_t product(const std::vector<item_t, rest_t...>& _container, const size_t _first, const size_t _last) {
		XERUS_REQUIRE(_first <= _last && _last <= _container.size(), "Invalid range " << _first << "-" << _last << " given (Container size " << _container.size() << ")"); 
		return std::accumulate(_container.begin()+_first, _container.begin()+_last, item_t(1), std::multiplies<item_t>());
	}
	
	
	///@brief: Erases all elements specified by @a _rule from the container @a _container.
	template<class rule_t, template<class, class...> class container_t, class item_t, class... rest_t>
	inline void erase(container_t<item_t, rest_t...>& _container, const rule_t& _rule) {
		_container.erase(std::remove_if(_container.begin(), _container.end(), _rule), _container.end());
	}
} } // namespaces xerus::misc



