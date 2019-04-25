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
 * @brief Implementation of the TensorNetwork::TensorNode class.
 */

#include <xerus/tensorNetwork.h>
#include <xerus/tensor.h>

namespace xerus {

    TensorNetwork::TensorNode::TensorNode() : erased(true) { }

    TensorNetwork::TensorNode::TensorNode(const TensorNetwork::TensorNode&  _other) : tensorObject(_other.tensorObject ? new Tensor(*_other.tensorObject) : nullptr), neighbors(_other.neighbors), erased(_other.erased) { }

    TensorNetwork::TensorNode::TensorNode(      std::unique_ptr<Tensor>&& _tensorObject) : tensorObject(std::move(_tensorObject)), neighbors(), erased(false) {}

    TensorNetwork::TensorNode::TensorNode(std::unique_ptr<Tensor>&& _tensorObject, std::vector<Link> _neighbors) : tensorObject(std::move(_tensorObject)), neighbors(std::move(_neighbors)), erased(false) {}

    TensorNetwork::TensorNode::~TensorNode() = default;

    TensorNetwork::TensorNode& TensorNetwork::TensorNode::operator=(const TensorNetwork::TensorNode&  _other) {
        if(_other.tensorObject) {
            if(tensorObject) {
                *tensorObject = *_other.tensorObject;
            } else {
                tensorObject.reset( new Tensor(*_other.tensorObject));
            }
        } else {
            tensorObject.reset();
        }
        neighbors = _other.neighbors;
        erased = _other.erased;
        return *this;
    }

    TensorNetwork::TensorNode& TensorNetwork::TensorNode::operator=( TensorNetwork::TensorNode&& _other) noexcept {
        tensorObject = std::move(_other.tensorObject);
        neighbors = std::move(_other.neighbors);
        erased = _other.erased;
        return *this;
    }

    TensorNetwork::TensorNode TensorNetwork::TensorNode::strippped_copy() const {
        return TensorNetwork::TensorNode(std::unique_ptr<Tensor>(), neighbors);
    }

    size_t TensorNetwork::TensorNode::size() const noexcept {
        size_t s = 1;
        for (const Link &l : neighbors) {
            s *= l.dimension;
        }
        return s;
    }

    size_t TensorNetwork::TensorNode::order() const noexcept {
        return neighbors.size();
    }

    void TensorNetwork::TensorNode::erase() noexcept {
        erased = true;
        neighbors.clear();
        tensorObject.reset();
    }
} // namespace xerus
