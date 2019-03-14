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
 * @brief Header file for macros that encapsulate SFINAE functionality.
 */

#pragma once

#include <type_traits>

/**
 * @def XERUS_ADD_MOVE(newTypeName, oldTypeName)
 * @brief Adds a template arguments whithin an existing template argument list, which can be of only one specified class, but allows & and && types.
 */
#define XERUS_ADD_MOVE(newTypeName, ...) class newTypeName, typename std::enable_if<std::is_base_of<__VA_ARGS__, typename std::decay<newTypeName>::type>::value, int>::type = 0


/**
 * @def XERUS_GENERATE_HAS_MEMBER(member)
 * @brief Macro to create a template class that checks for the existence of member functions. To be used in other template definitions in a SFINAE fashion.
 */
#if __GNUC__ > 4 || defined(__clang__)
	// template void
	template<class...> using void_t = void;


	#define XERUS_GENERATE_HAS_FUNCTION(function)\
	namespace sfinae {\
		template<class, class, class = void>\
		struct has_##function : std::false_type {};\
		\
		template<class clas, class arg>\
		struct has_##function<clas, arg, void_t<decltype(std::declval<clas>().function(std::declval<arg>()))>> : std::true_type {};\
	}\
	\
	
	
	// TODO Gives a false positive if a function can be called by implicit casting!
	#define XERUS_GENERATE_EXISTS_FUNCTION(function)\
	namespace sfinae {\
        template<class, class = void>\
        struct exists_##function : std::false_type {};\
        \
        template<class arg>\
        struct exists_##function<arg, void_t<decltype(function(std::declval<arg>()))>> : std::true_type {};\
	}\
	\

#else
	#define XERUS_GENERATE_HAS_MEMBER(member)											   \
	\
	template < class T >															  \
	class HasMember_##member														  \
	{																				 \
	public:																		  \
		typedef char (& yes)[1];							\
		typedef char (& no)[2];							\
		\
		template <typename C> static yes check(decltype(&C::member));		\
		template <typename> static no check(...);					\
		\
		static constexpr bool RESULT = sizeof(check<T>(0)) == sizeof(yes);		\
		};																				\
		\
		template < class T >															  \
		struct has_member_##member														\
		: public std::integral_constant<bool, HasMember_##member<T>::RESULT>			  \
		{ };																			  \

#endif
