// Xerus - A General Purpose Tensor Library
// Copyright (C) 2014-2015 Benjamin Huber and Sebastian Wolf. 
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

#ifndef XERUS_H
#define XERUS_H

    #define MISC_NAMESPACE xerus

    // The misc stuff needed by xerus
    #include "xerus/misc/standard.h"
    #include "xerus/misc/namedLogger.h"
    #include "xerus/misc/missingFunctions.h"
    #include "xerus/misc/stringUtilities.h"
    #include "xerus/misc/sfinae.h"
    #include "xerus/misc/blasLapackWrapper.h"
    #include "xerus/misc/selectedFunctions.h"
    #include "xerus/misc/callStack.h"
    #include "xerus/misc/simpleNumerics.h"
    #include "xerus/misc/timeMeasure.h"
    #include "xerus/misc/performanceAnalysis.h"

    // File which sets the custom log levels
    #include "xerus/tensorLogger.h"

    // All the xerus headers
    #include "xerus/index.h"
    #include "xerus/indexedTensorReadOnly.h"
    #include "xerus/indexedTensorWritable.h"
    #include "xerus/indexedTensor.h"
    #include "xerus/indexedTensorMoveable.h"
    #include "xerus/indexedTensorList.h"
    #include "xerus/tensor.h"
    #include "xerus/fullTensor.h"
    #include "xerus/sparseTensor.h"
    #include "xerus/cs_wrapper.h"
    #include "xerus/sparseTimesFullContraction.h"
    #include "xerus/sparseTimesFullContraction.h"
    #include "xerus/indexedTensor_tensor_operators.h"
    #include "xerus/indexedTensor_tensor_factorisations.h"
    #include "xerus/tensorNode.h"
    #include "xerus/tensorNetwork.h"
    #include "xerus/indexedTensor_TN_operators.h"
    #include "xerus/contractionHeuristic.h"
    #include "xerus/ttNetwork.h"
    #include "xerus/algorithms/als.h"

	
#endif
