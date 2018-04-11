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
 * @brief Header file for the TTStack class.
 */

#pragma once

#include "misc/check.h"

#include "tensorNetwork.h"
#include "indexedTensorMoveable.h"

namespace xerus {
	template<bool isOperator> class TTNetwork;
	
	namespace internal {
		template<bool isOperator>
		///@brief Internal class used to represent stacks of (possibly multiply) applications of TTOperators to either a TTTensor or TTOperator.
		class TTStack final : public TensorNetwork {
		public:
			///@brief The number of external links in each node, i.e. one for TTTensors and two for TTOperators.
			static constexpr const size_t N = isOperator?2:1;
		
			const bool cannonicalization_required;
			
			const size_t futureCorePosition;
			
			explicit TTStack(const bool _canno, const size_t _corePos = 0);
			
			TTStack(const TTStack&  _other) = default;
			
			TTStack(      TTStack&& _other) = default;
			
			virtual TensorNetwork* get_copy() const override;
			
			TTStack& operator= (const TTStack&  _other) = delete;
			
			TTStack& operator= (      TTStack&& _other) = delete;
			
			explicit operator TTNetwork<isOperator>();
			
			virtual void operator*=(const value_t _factor) override;
			
			virtual void operator/=(const value_t _divisor) override;
			
			
			/*- - - - - - - - - - - - - - - - - - - - - - - - - - Operator specializations - - - - - - - - - - - - - - - - - - - - - - - - - - */
			virtual void specialized_evaluation(IndexedTensorWritable<TensorNetwork>&& , IndexedTensorReadOnly<TensorNetwork>&&) override;
			
			virtual bool specialized_contraction(std::unique_ptr<IndexedTensorMoveable<TensorNetwork>>& _out, IndexedTensorReadOnly<TensorNetwork>&& _me, IndexedTensorReadOnly<TensorNetwork>&& _other) const override;
			
			virtual bool specialized_sum(std::unique_ptr<IndexedTensorMoveable<TensorNetwork>>& _out, IndexedTensorReadOnly<TensorNetwork>&& _me, IndexedTensorReadOnly<TensorNetwork>&& _other) const override;
			
			static void contract_stack(IndexedTensorWritable<TensorNetwork>&& _me);
			
			virtual value_t frob_norm() const override;
		};
	}
}
