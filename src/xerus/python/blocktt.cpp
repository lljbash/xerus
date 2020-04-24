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
 * @brief Definition of the TT-Network python bindings.
 */


#define NO_IMPORT_ARRAY
#include "misc.h"

using namespace internal;

void expose_blocktt() {
	VECTOR_TO_PY(BlockTT, "BlockTTVector");

	class_<BlockTT>("BlockTT")
		.def(init<const std::vector<size_t>&, const std::vector<size_t>&, const size_t,const size_t>())
		.def(init<const TTTensor &,const size_t,const size_t >())
		.def("get_component", &BlockTT::get_component, return_value_policy<copy_const_reference>())
		.def("set_component", &BlockTT::set_component)
		.def("ranks", &BlockTT::ranks)
		.def("rank", &BlockTT::rank)
		.def("num_components", &BlockTT::num_components)
		.def("get_core", &BlockTT::get_core)
		.def("get_average_core", &BlockTT::get_average_core)
		.def("get_average_tt", &BlockTT::get_average_tt)
		.def("order", &BlockTT::order)
		.def("move_core", static_cast<void (BlockTT::*)(const size_t,const double,const size_t)>(&BlockTT::move_core),
			(arg("position"), arg("epsilon")=EPSILON, arg("maxRank")=std::numeric_limits<size_t>::max())
		)
		.def("average_core", &BlockTT::average_core)
		.def("all_entries_valid", &BlockTT::all_entries_valid)
		.def("frob_norm", &BlockTT::frob_norm)
		.def("dofs", &BlockTT::dofs)

		.def("move_core_left", &BlockTT::move_core_left)
		.def("move_core_right", &BlockTT::move_core_right)
	;

	def("frob_norm", static_cast<value_t (*)(const BlockTT&)>(&frob_norm));
}
