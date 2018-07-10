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
 * @brief Implementation of the TensorNetwork::Link class.
 */

#include <xerus/tensorNetwork.h>

namespace xerus {
    TensorNetwork::Link::Link(const size_t _other, const size_t _indexPos, const size_t _dim, const bool _external) noexcept
        : other(_other), indexPosition(_indexPos), dimension(_dim), external(_external) {}

    bool TensorNetwork::Link::links(const size_t _other) const noexcept { return !external && other == _other; }

    /*- - - - - - - - - - - - - - - - - - - - - - - - - - External functions - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/

    std::ostream &operator<<(std::ostream &_out, const xerus::TensorNetwork::Link &_rhs) {
        _out << "L{";
        if (_rhs.external) {
            _out << "ext";
        } else {
            _out << _rhs.other;
        }
        _out << " (" << _rhs.indexPosition << "), dim " << _rhs.dimension << "}";
        return _out;
    }
} // namespace xerus
