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

#include <xerus/indexedTensorMoveable.h>
#include <xerus/tensor.h>

namespace xerus {
    /*
    template<>template<>
    IndexedTensorMoveable<Tensor>::IndexedTensorMoveable(const IndexedTensorReadOnly<Tensor> &  _other) : 
        IndexedTensorWritable<Tensor>(_other.tensorObjectReadOnly->get_copy(), _other.indices, true) { }
    
    template<>template<>
    IndexedTensorMoveable<Tensor>::IndexedTensorMoveable(      IndexedTensorReadOnly<Tensor> && _other) : 
        IndexedTensorWritable<Tensor>(_other.tensorObjectReadOnly->get_copy(), std::move(_other.indices), true) { }*/
} // namespace xerus