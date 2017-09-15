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
 * @brief Header file for the TTBlock class.
 */

#pragma once

#include "tensor.h"
#include "tensorNetwork.h"
#include "ttNetwork.h"

namespace xerus { namespace internal {
	/**
	 * @brief Specialized TensorNetwork class used to represent a BlockTT
	 */
	class BlockTT final : public TensorNetwork {
	public:
		/// @brief Flag indicating whether the TTNetwork is canonicalized.
		bool canonicalized;
		
		/**
		 * @brief The position of the core.
		 * @details If canonicalized is TRUE, corePosition gives the position of the core tensor. All components
		 * with smaller index are then left-orthogonalized and all components with larger index right-orthogonalized.
		 */
		size_t corePosition;		
		
		
		/*- - - - - - - - - - - - - - - - - - - - - - - - - - Constructors - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
		/** 
		 * @brief Constructs an order zero TTNetwork.
		 * @details This is an empty TensorNetwork. Internally the network contains one order zero node with entry zero.
		 */
		explicit BlockTT() = delete;
		
		
		///@brief TTNetworks are default copy constructable.
		BlockTT(const BlockTT & _cpy) = default;
		
		
		///@brief TTNetworks are default move constructable.
		BlockTT(      BlockTT&& _mov) noexcept = default;
		
		
		/** 
		 * @brief Constructs a BlockTT from the given TTTensor.
		 */
		explicit BlockTT(const TTTensor& _tensor, const size_t _blockPosition);
		
		
		
		/*- - - - - - - - - - - - - - - - - - - - - - - - - - Standard Operators - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
		///@brief TTNetworks are default assignable.
		BlockTT& operator=(const BlockTT&  _other) = default;
		
		
		///@brief TTNetworks are default move-assignable.
		BlockTT& operator=(      BlockTT&& _other) = default;
		
		
		/*- - - - - - - - - - - - - - - - - - - - - - - - - - Miscellaneous - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
		public:
			
			/** 
			* @brief Complete access to a specific component of the TT decomposition.
			* @note This function will not update rank and external dimension informations if it is used to set a component.
			* @details This function gives complete access to the components, only intended for internal use.
			* @param _idx index of the component to access.
			* @returns a reference to the requested component.
			*/
			Tensor& component(const size_t _idx);
			
			
			/** 
			* @brief Read access to a specific component of the TT decomposition.
			* @details This function should be used to access the components, instead of direct access via
			* nodes[...], because the implementation does not store the first component in nodes[0] but rather as
			* nodes[1] etc. nodes[0] is an order one node with dimension one only used to allow the first component
			* to be an order three tensor.
			* @param _idx index of the component to access.
			* @returns a const reference to the requested component.
			*/
			const Tensor& get_component(const size_t _idx) const;
			
			
			/** 
			* @brief Sets a specific component of the TT decomposition.
			* @details This function also takes care of adjusting the corresponding link dimensions and external dimensions
			* if needed. However this might still leave the TTNetwork in an invalid if the rank is changed. In this case it
			* is the callers responsibility to also update the other component tensors consistently to account for that rank
			* change.
			* @param _idx index of the component to set.
			* @param _T Tensor to use as the new component tensor.
			*/
			void set_component(const size_t _idx, Tensor _T);
			
			
			/** 
			* @brief Gets the ranks of the TTNetwork.
			* @return A vector containing the current ranks.
			*/
			std::vector<size_t> ranks() const;
			
			
			/** 
			* @brief Gets the rank of a specific egde of the TTNetwork.
			* @param _i Position of the edge in question.
			* @return The current rank of edge _i.
			*/
			size_t rank(const size_t _i) const;
			
			
			/** 
			* @brief Move the core to a new position.
			*/
			void move_core(const size_t _position, const double _eps=EPSILON);
			
			
			/**
				* @brief stores @a _pos as the current core position without verifying of ensuring that this is the case
				* @details this is particularly useful after constructing an own TT tensor with set_component calls
				* as these will assume that all orthogonalities are destroyed
				*/
			void assume_core_position(const size_t _pos);
			
			
			virtual value_t frob_norm() const override;
			
			
			/** 
			* @brief 
			*/
			virtual void require_correct_format() const override;
			
	};
} }
