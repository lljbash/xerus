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
 * @brief Header file for comfort functions and macros that should not be exported in the library.
 * @note this may only be used in cpp files and must NOT be included in any library header!
 */

#pragma once

#include <type_traits>
#include <memory>

#include "containerOutput.h"
#include "namedLogger.h"
#include "check.h"

// macro shorthands
#define REQUIRE 		XERUS_REQUIRE
#define CHECK			XERUS_CHECK
#define IF_CHECK		XERUS_IF_CHECK
#define IF_NO_CHECK		XERUS_IF_NO_CHECK
#define INTERNAL_CHECK	XERUS_INTERNAL_CHECK
#define LOG				XERUS_LOG
#define LOG_SHORT		XERUS_LOG_SHORT
#define LOG_ONCE		XERUS_LOG_ONCE
#define IS_LOGGING		XERUS_IS_LOGGING
#define SET_LOGGING		XERUS_SET_LOGGING



namespace std {
#if !defined(__cplusplus) || __cplusplus < 201402L
	template<typename T, typename... Args>
	std::unique_ptr<T> make_unique(Args&&... args) {
	    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
	}
#endif
	
	
	using ::xerus::misc::operator<<; // for std::ostream << std::vector etc.
	
	//TODO No good solution, as this requires the use of std::advance, which may be less efficient than the +operator for random access iterators. (Need concepts)
	///@brief Add a + operator for iterators and size_t to avoid signed/unsigned errors.
	template<class IteratorType, 
		typename std::enable_if<
			std::is_same<typename std::iterator_traits<IteratorType>::difference_type, long>::value
			&& std::is_class<IteratorType>::value
		, bool>::type = true>
	IteratorType operator+(const IteratorType& _iterator, const size_t _add) {
		IteratorType itr = _iterator;
		std::advance(itr, _add);
		return itr;
	}
	
	
	///@brief: Concatenates two given cointainers.
	template<template<class, class...> class container_t, class item_t, class... rest_t>
	container_t<item_t, rest_t...> operator |(const container_t<item_t, rest_t...> & _left, const container_t<item_t, rest_t...> & _right) {
		container_t<item_t, rest_t...> both(_left);
		both.insert(both.end(), _right.begin(), _right.end());
		return both;
	}
}
