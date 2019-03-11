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
 * @brief Implementation of the (indexed) Tensor evaluation (ie. generlized transpositions).
 */

#include <xerus/basic.h>
#include <xerus/index.h>
#include <xerus/misc/check.h>
#include <xerus/tensor.h>
 
#include <memory>
#include <xerus/misc/basicArraySupport.h>
 
#include <xerus/misc/containerSupport.h>
#include <xerus/misc/performanceAnalysis.h>
#include <xerus/misc/internal.h>

namespace xerus {

	void increase_indices(const size_t _i, value_t*& _outPosition, const std::vector<size_t>& _stepSizes, const std::vector<size_t>& _baseDimensions ) {
		size_t index = _stepSizes.size()-1;
		_outPosition += _stepSizes[index];
		size_t multStep = _baseDimensions[index];
		while(_i%multStep == 0) {
			_outPosition -= _baseDimensions[index]*_stepSizes[index]; // "reset" current index to 0
			--index; // Advance to next index
			_outPosition += _stepSizes[index]; // increase next index
			multStep *= _baseDimensions[index]; // next stepSize
		}
	}
	
	/**
	 * @brief: Performs a simple reshuffle. Much less powerfull then a full evaluate, but more efficient.
	 * @details @a _shuffle shall be a vector that gives for every old index, its new position.
	 */
	void reshuffle(Tensor& _out, const Tensor& _base, const std::vector<size_t>& _shuffle) {
		IF_CHECK(
			INTERNAL_CHECK(_shuffle.size() == _base.degree(), "IE");
			for(size_t x = 0; x < _base.degree(); ++x) {
				REQUIRE(_shuffle[x] < _base.degree(), _shuffle[x] << " is no valid new position!");
				REQUIRE(misc::count(_shuffle, x) == 1, x << " illegally appeared twice.");
			}
		)
		
		// Count how many indices in the back do not change and calculate blockSize
		long lastShuffled = -1;
		for(size_t i = 0; i < _shuffle.size(); ++i) {
			if(_shuffle[i] != i) { lastShuffled = long(i) ; }
		}
		
		const size_t numToShuffle = size_t(lastShuffled+1);
		const size_t lastToShuffle = _base.dimensions.size(); // NOTE inlining this into the next line causes an internal segfault in gcc 4.8.1
		const size_t blockSize = misc::product(_base.dimensions, numToShuffle, lastToShuffle);
		
		if(blockSize == _base.size) { // No actual reshuffling
			_out = _base;
			return;
		}
		
		std::vector<size_t> outDimensions(_base.degree());
		for( size_t i = 0; i < _base.degree(); ++i ) {
			outDimensions[_shuffle[i]] = _base.dimensions[i];
		}
		
		std::vector<size_t> stepSizes(numToShuffle);
		for( size_t i = 0; i < numToShuffle; ++i ) {
			stepSizes[i] = misc::product(outDimensions, _shuffle[i]+1, numToShuffle)*blockSize;
		}
		
		// Rescue _base from reset in case _out and _base coincide
		std::unique_ptr<Tensor> rescueSlot;
		const Tensor* usedBase;
		if(&_out == &_base) {
			rescueSlot.reset(new Tensor(std::move(_out)));
			usedBase = rescueSlot.get();
		} else {
			usedBase = &_base;
		}
		
		_out.reset(std::move(outDimensions), usedBase->representation, Tensor::Initialisation::None);
		
		if(usedBase->is_dense()) {
			const size_t numBlocks = misc::product(usedBase->dimensions, 0, numToShuffle);
			value_t* outPosition = _out.override_dense_data();
			const value_t* basePosition = usedBase->get_unsanitized_dense_data();
			
			if(blockSize == 1) {
				*outPosition = *basePosition;
				for(size_t b = 1; b < numBlocks; ++b) {
					increase_indices(b, outPosition, stepSizes, usedBase->dimensions);
					*outPosition = *(basePosition + b*blockSize);
				}
			} else {
				misc::copy(outPosition, basePosition, blockSize);
				for(size_t b = 1; b < numBlocks; ++b) {
					increase_indices(b, outPosition, stepSizes, usedBase->dimensions);
					misc::copy(outPosition, basePosition + b*blockSize, blockSize);
				}
			}
			_out.factor = usedBase->factor;
			
		} else {
			const std::map<size_t, value_t>& baseEntries = usedBase->get_unsanitized_sparse_data();
			std::map<size_t, value_t>& outEntries = _out.override_sparse_data();
			
			for(const auto& entry : baseEntries) {
				size_t basePosition = entry.first;
				size_t position = basePosition%blockSize;
				basePosition /= blockSize;
				
				for(size_t i = numToShuffle; i > 0; --i) {
					position += (basePosition%usedBase->dimensions[i-1])*stepSizes[i-1];
					basePosition /= usedBase->dimensions[i-1];
				}
				outEntries.emplace(position, usedBase->factor*entry.second);
			}
		}
	}
	
	Tensor reshuffle(const Tensor& _base, const std::vector<size_t>& _shuffle) {
		Tensor result;
		reshuffle(result, _base, _shuffle);
		return result;
	}
	
	namespace internal {
		
		inline void increase_indices(const size_t _i, const value_t*& _oldPosition, const std::vector<size_t>& _steps, const std::vector<size_t>& _multDimensions) {
			size_t index = _steps.size()-1;
			_oldPosition += _steps[index];
			size_t multStep = _multDimensions[index];
			while(_i%multStep == 0) {
				_oldPosition -= _multDimensions[index]*_steps[index]; // "reset" current index to 0
				--index; // Advance to next index
				_oldPosition += _steps[index]; // increase next index
				multStep *= _multDimensions[index]; // next stepSize
			}
		}

		inline void sum_traces(	value_t* const _newPosition,
								const value_t* _oldPosition,
								const std::vector<size_t>& _doubleSteps,
								const std::vector<size_t>& _doubleMultDimensions,
								const size_t _numSummations) {
			*_newPosition = *_oldPosition;
			for(size_t k = 1; k < _numSummations; ++k) {
				increase_indices(k, _oldPosition, _doubleSteps, _doubleMultDimensions);
				*_newPosition += *_oldPosition;
			}
		}
		
		inline void sum_traces(	value_t* const _newPosition,
								const value_t* _oldPosition,
								const std::vector<size_t>& _doubleSteps,
								const std::vector<size_t>& _doubleMultDimensions,
								const size_t _numSummations,
								const size_t _orderedIndicesMultDim ) {
			misc::copy(_newPosition, _oldPosition, _orderedIndicesMultDim);
			for(size_t k = 1; k < _numSummations; ++k) {
				increase_indices(k, _oldPosition, _doubleSteps, _doubleMultDimensions);
				misc::add(_newPosition, _oldPosition, _orderedIndicesMultDim);
			}
		}
		
		
		inline size_t get_position(const std::pair<size_t, value_t>& _entry,
							const size_t* const _baseIndexDim,
							const size_t* const _divisors,
							const size_t* const _attributes,
							const size_t numIndices ) {
			size_t position = 0;
			for(size_t i = 0; i < numIndices; ++i) {
				position += ((_entry.first/_divisors[i])%_baseIndexDim[i])*_attributes[i];
			}
			return position;
		}

		inline bool check_position(size_t & _position,
							const std::pair<size_t, value_t>& _entry,
							const size_t* const _baseIndexDim,
							const size_t* const _divisors,
							const size_t* const _attributes,
							const bool* const _fixedFlags,
							const bool* const _traceFlags,
							const size_t numIndices ) {
			_position = 0;
			// Process each index
			for(size_t i = 0; i < numIndices; ++i) {
				const size_t indexPosition = (_entry.first/_divisors[i])%_baseIndexDim[i];
				
				// The position is only changed if the index is not fixed...
				if(_fixedFlags[i]) {
					if(indexPosition != _attributes[i]) {
						return false; // The indexPosition differs from the fixed position => Do not add this value
					}
				// ... and not part of a trace.
				} else if(_traceFlags[i]) {
					if(indexPosition != (_entry.first/_divisors[_attributes[i]])%_baseIndexDim[_attributes[i]]) {
						return false; // This entry is not on the diagonal of the trace => Do not add this value
					}
				} else {
					_position += indexPosition*_attributes[i];
				}
			}
			return true;
		}

		inline std::vector<size_t> get_dimension_vector(const std::vector<Index>& _indices) {
			std::vector<size_t> dimensions;
			dimensions.reserve(_indices.size());
			for(const Index& idx : _indices) {
				dimensions.emplace_back(idx.dimension());
			}
			return dimensions;
		}
		
		inline std::vector<size_t> get_step_sizes(const std::vector<Index>& _indices) {
			std::vector<size_t> stepSizes(_indices.size());
			if(!_indices.empty()) {
				stepSizes.back() = 1;
				for(size_t i = stepSizes.size(); i > 1; --i) {
					stepSizes[i-2] = stepSizes[i-1]*_indices[i-1].dimension();
				}
			}
			return stepSizes;
		}


		void evaluate(internal::IndexedTensorWritable<Tensor>&& _out, internal::IndexedTensorReadOnly<Tensor>&& _base) {
			// Assign the indices
			_base.assign_indices();
			_out.assign_indices();
			
			
			// Assign the dimensions
			_base.assign_index_dimensions();
			_out.assign_index_dimensions();
			
			#ifndef XERUS_DISABLE_RUNTIME_CHECKS // Performe complete check whether the input is valid
				// Check base indices
				for(size_t i = 0; i < _base.indices.size(); ++i) {
					// If the index is fixed we don't expect to find it anywhere
					if(_base.indices[i].fixed()) {
						REQUIRE(_base.indices[i].span == 1, "Fixed indices must have span one. Indices are: " << _base.indices << ", total should be " << _base.indices.size() << ". The problem is: " << _base.indices[i] << " -- " << _base.indices[i].fixed());
						continue;
					}
					
					// Try to find index in _out
					size_t j = 0;
					while(j < _out.indices.size() && _base.indices[i] != _out.indices[j]) { ++j; }
					if(j < _out.indices.size()) {
						REQUIRE(_base.indices[i].dimension() == _out.indices[j].dimension(), "The indexDimensions in the target and base of evaluation must coincide. Here " << _base.indices[i].dimension() << "!=" << _out.indices[j].dimension() << ". For index " << _base.indices[i] << " == " << _out.indices[j]);
						REQUIRE(_base.indices[i].span == _out.indices[j].span, "The indexSpans in the target and base of evaluation must coincide.");
						REQUIRE(_base.indices[i].open(), "Indices appearing in the target of evaluation must not be part of a trace nor be fixed. Base: " << _base.indices << " Out: " << _out.indices);
						continue;
					}
					
					// Try to find index a second time in base
					j = 0;
					while(j < _base.indices.size() && (i == j || _base.indices[i] != _base.indices[j])) { ++j; }
					REQUIRE(j < _base.indices.size(), "All indices of evalutation base must either be fixed, appear in the target or be part of a trace. Base: " << _base.indices << " Out: " << _out.indices);
					REQUIRE(misc::count(_base.indices, _base.indices[i]) == 2, "Indices must appear at most two times. Base: " << _base.indices << " Out: " << _out.indices);
					REQUIRE(_base.indices[i].dimension() == _base.indices[j].dimension(), "The indexDimensions of two traced indices must conince.");
					REQUIRE(_base.indices[i].span == 1 && _base.indices[j].span == 1, "The indexSpans of traced indices must be one (It is ambigious what a trace of span 2 indices is meant to be).");
				}
				
				// Check out indices
				for(size_t i = 0; i < _out.indices.size(); ++i) {
					REQUIRE(_out.indices[i].open(),  "Traces and fixed indices are not allowed in the target of evaluation. Base: " << _base.indices << " Out: " << _out.indices);
					REQUIRE(misc::count(_base.indices, _out.indices[i]) == 1, "Every index of the target must appear exactly once in the base of evaluation. Base: " << _base.indices << " Out: " << _out.indices);
				}
			#endif
			
			// If there is no index reshuffling, the evalutation is trivial (NOTE must be after index assignment so span zero indices are erased. Otherwise empty index arrays may occur)
			if(_base.indices == _out.indices) {
				*_out.tensorObject = *_base.tensorObjectReadOnly;
				return; // We are finished here
			}
			
			// Extract base index dimensions and stepSizes
			const std::vector<size_t> baseIndexStepSizes(get_step_sizes(_base.indices));
			const std::vector<size_t> baseIndexDimensions(get_dimension_vector(_base.indices));
			
			//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - Full => Full   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
			if(_base.tensorObjectReadOnly->is_dense()) {
				// Extract out index dimensions
				const std::vector<size_t> outIndexDimensions(get_dimension_vector(_out.indices));
				
				// Count how many indices in the back are already ordered (We know that base has at least as many indices as out)
				size_t numOrderedIndices;
				for(numOrderedIndices = 0; numOrderedIndices < _out.indices.size() && _base.indices[_base.indices.size()-1-numOrderedIndices] == _out.indices[_out.indices.size()-1-numOrderedIndices]; ++numOrderedIndices) { }
				
				const size_t orderedIndexDim = baseIndexStepSizes[_base.indices.size()-numOrderedIndices-1];
				
				std::vector<size_t> stepSizes(_out.indices.size()-numOrderedIndices); // How much we have to move in base when the i-th index of _out increases
				
				size_t fixedIndexOffset = 0; // Fixed offset in base due to fixed indices
				
				std::vector<size_t> traceStepSizes; // How much we have to move in base if the index of the i-th trace is increased
				std::vector<size_t> traceDimensions; // The dimensions of the traces
				size_t totalTraceDim = 1; // Total number of summantions, i.e. Product of all trace dimensions
				
				// Calculate stepSizes for our tensor. We will march in steps of orderedIndicesMultDim in _out.data
				for(size_t i = 0; i < _base.indices.size()-numOrderedIndices; ++i) {
					// First try to find the index in _out (if it is not contained it must be fixed or part of a trace)
					size_t outPos = 0;
					while(outPos < _out.indices.size() && _base.indices[i] != _out.indices[outPos]) { ++outPos; }
					
					if(outPos < _out.indices.size()) { // If we found it we are basically finished
						stepSizes[outPos] = baseIndexStepSizes[i];
					} else if(_base.indices[i].fixed()) { // One reason for an index to not be in _out is to be fixed
						fixedIndexOffset += _base.indices[i].fixed_position()*baseIndexStepSizes[i];
					} else { // If the Index is not fixed, then it has to be part of a trace
						for(size_t j = i+1; j < _base.indices.size()-numOrderedIndices; ++j) {
							if(_base.indices[i] == _base.indices[j]) {
								traceStepSizes.emplace_back(baseIndexStepSizes[i]+baseIndexStepSizes[j]);
								traceDimensions.emplace_back(_base.indices[i].dimension());
								totalTraceDim *= _base.indices[i].dimension();
								break;
							}
						}
					}
				}
				
				// Start performance analysis for low level part
				XERUS_PA_START;
				
				// Get pointers to the data and delay the deletion of the base data in case _out and _base coincide
				const value_t* oldPosition = _base.tensorObjectReadOnly->get_unsanitized_dense_data()+fixedIndexOffset;
				std::shared_ptr<value_t> delaySlot;
				if(_base.tensorObjectReadOnly == _out.tensorObjectReadOnly) { delaySlot = _out.tensorObject->get_internal_dense_data(); }
				value_t* const newPosition = _out.tensorObject->override_dense_data();
				
				// Get specific sizes
				const size_t numSteps = _out.tensorObject->size/orderedIndexDim;
				
				if(orderedIndexDim == 1) { // We have to copy/add single entries
					if(totalTraceDim == 1) { // We don't need to sum any traces
						newPosition[0] = *oldPosition;
						for(size_t i = 1; i < numSteps; ++i) {
							increase_indices(i, oldPosition, stepSizes, outIndexDimensions);
							newPosition[i] = *oldPosition;
						}
					} else { // We have to add traces
						sum_traces(newPosition, oldPosition, traceStepSizes, traceDimensions, totalTraceDim);
						for(size_t i = 1; i < numSteps; ++i) {
							increase_indices(i, oldPosition, stepSizes, outIndexDimensions);
							sum_traces(newPosition+i, oldPosition, traceStepSizes, traceDimensions, totalTraceDim);
						}
					}
				} else { // We can copy/add larger blocks
					if(totalTraceDim == 1) { // We don't need to sum any traces
						misc::copy(newPosition, oldPosition, orderedIndexDim);
						for(size_t i = 1; i < numSteps; ++i) {
							increase_indices(i, oldPosition, stepSizes, outIndexDimensions);
							misc::copy(newPosition + i*orderedIndexDim, oldPosition, orderedIndexDim);
						}
					} else { // We have to add traces
						sum_traces(newPosition, oldPosition, traceStepSizes, traceDimensions, totalTraceDim, orderedIndexDim);
						for(size_t i = 1; i < numSteps; ++i) {
							increase_indices(i, oldPosition, stepSizes, outIndexDimensions);
							sum_traces(newPosition + i*orderedIndexDim, oldPosition, traceStepSizes, traceDimensions, totalTraceDim, orderedIndexDim);
						}
					}
				}
				XERUS_PA_END("Evaluation", "Full->Full", misc::to_string(_base.tensorObjectReadOnly->dimensions)+" ==> " + misc::to_string(_out.tensorObject->dimensions));
				
				
				// Propagate the constant factor, since we don't apply it for dense Tensors
				_out.tensorObject->factor = _base.tensorObjectReadOnly->factor;
			}
			//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - Sparse => Sparse  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
			else if(_base.tensorObjectReadOnly->is_sparse()) {
				#define VLA(T, name) auto name##_store = xerus::misc::make_unique_array(new T); const auto & name = name##_store.get();
				VLA(bool[_base.indices.size()]  , fixedFlags); // Flag for each index indicating whether the index is fixed
				VLA(bool[_base.indices.size()]  , traceFlags); // Flag for each index indicating whether the index is part of a trace
				VLA(size_t[_base.indices.size()], attributes);  // Either the factor in _out, the value of an fixed index or the position of the other part of a trace
				#undef VLA
				bool peacefullIndices = true;
				
				const std::vector<size_t> outIndexStepSizes(get_step_sizes(_out.indices));
				
				// Process base indices
				for(size_t i = 0; i < _base.indices.size(); ++i) {
					// Check whether the index is fixed
					if(_base.indices[i].fixed()) {
						fixedFlags[i] = true;
						traceFlags[i] = false;
						attributes[i] = _base.indices[i].fixed_position();
						peacefullIndices = false;
						continue;
					}
					
					// Try to find the index in out
					size_t outPos = 0;
					while(outPos < _out.indices.size() && _out.indices[outPos] != _base.indices[i]) { ++outPos; }
					if(outPos < _out.indices.size()) {
						fixedFlags[i] = false;
						traceFlags[i] = false;
						attributes[i] = outIndexStepSizes[outPos];
						continue;
					}
					
					// If the index is not in out and not fixed then it has to be part of a trace
					fixedFlags[i] = false;
					traceFlags[i] = true;
					peacefullIndices = false;
					for(attributes[i] = 0; _base.indices[i] != _base.indices[attributes[i]] || attributes[i] == i; ++attributes[i]) { }
				}
				
				// Get direct acces to the entries and delay deletion of _base data in case _base and _out coincide
				const std::map<size_t, value_t>& baseEntries = _base.tensorObjectReadOnly->get_unsanitized_sparse_data();
				std::shared_ptr<std::map<size_t, value_t>> delaySlot;
				if(_base.tensorObjectReadOnly == _out.tensorObjectReadOnly) { delaySlot = _out.tensorObject->get_internal_sparse_data(); }
				std::map<size_t, value_t>& outEntries = _out.tensorObject->override_sparse_data(); // Takes care that no entries are present
				
				const value_t factor = _base.tensorObjectReadOnly->factor;
				
				// Start performance analysis for low level part
				XERUS_PA_START;
				
				if(peacefullIndices) {
					for(const auto& entry : baseEntries) {
						outEntries.emplace(get_position(entry, baseIndexDimensions.data(), baseIndexStepSizes.data(), attributes, _base.indices.size()), factor*entry.second);
					}
				} else {
					size_t newPosition;
					for(const auto& entry : baseEntries) {
						if(check_position(newPosition, entry, baseIndexDimensions.data(), baseIndexStepSizes.data(), attributes, fixedFlags, traceFlags, _base.indices.size())) {
							outEntries[newPosition] += factor*entry.second;
						}
					}
				}
				XERUS_PA_END("Evaluation", "Sparse->Sparse", misc::to_string(_base.tensorObjectReadOnly->dimensions)+" ==> " + misc::to_string(_out.tensorObjectReadOnly->dimensions));
			}
		}
	} // namespace internal
} // namespace xerus
