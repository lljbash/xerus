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
 * @brief Implementation of the custom new and delete operators based on a bucket-allocator idea.
 */


#ifdef XERUS_REPLACE_ALLOCATOR

	#include <xerus/misc/allocator.h>
	// #include <dlfcn.h>
	#include <cstring>

	using xma = xerus::misc::AllocatorStorage;
	namespace xm = xerus::misc;

	thread_local xerus::misc::AllocatorStorage xm::astore;
	thread_local bool programIsRunning = true;

	xerus::misc::AllocatorStorage::AllocatorStorage() {
		for (unsigned long i=0; i<xma::NUM_BUCKETS; ++i) {
			allocCount[i]=0;
			maxAlloc[i]=0;
			currAlloc[i]=0;
		}
	}

	xerus::misc::AllocatorStorage::~AllocatorStorage(){ programIsRunning=false; }


	using mallocType = void *(*)(size_t);
	void* (*xerus::misc::r_malloc)(size_t) = &malloc;//(mallocType)dlsym(RTLD_NEXT, "malloc");
	using freeType = void (*)(void*);
	void (*xerus::misc::r_free)(void*) = &free;//(freeType)dlsym(RTLD_NEXT, "free");


	void xerus::misc::AllocatorStorage::create_new_pool() {
		uint8_t* newPool = static_cast<uint8_t*>(xerus::misc::r_malloc(xma::POOL_SIZE)); // TODO use mmap(...)
		uint8_t* startAddr = newPool+xma::BUCKET_SIZE - reinterpret_cast<uintptr_t>(newPool)%xma::ALIGNMENT;
		pools.emplace_back(newPool, startAddr);
	}

	void *myalloc(size_t n) {
		if (n >= xma::SMALLEST_NOT_CACHED_SIZE) {
			void *res = xerus::misc::r_malloc(n+xma::ALIGNMENT+1);
			
			// res is increased by at least 2 bytes (as alignmentOffset < xma::ALIGNMENT)
			res = static_cast<void*>(static_cast<uint8_t*>(res)+xma::ALIGNMENT+1);
			uintptr_t alignmentOffset = reinterpret_cast<uintptr_t>(res)%xma::ALIGNMENT;
			res = static_cast<void*>(static_cast<uint8_t*>(res) - alignmentOffset);
			
			// indicate non-bucket allocation
			*(static_cast<uint8_t*>(res)-1) = 0xFF;
			// and the alignmentOffset to be able to reconstruct the original res pointer
			*(static_cast<uint8_t*>(res)-2) = static_cast<uint8_t>(alignmentOffset);
			return res;
		} else {
			uint8_t numBucket = uint8_t( (n+1)/xma::BUCKET_SIZE );
			uint8_t* res;
			if (xm::astore.buckets[numBucket].empty()) {
				if (xm::astore.pools.empty() || xm::astore.pools.back().second + (numBucket+1)*xma::BUCKET_SIZE >= xm::astore.pools.back().first + xma::POOL_SIZE) {
					xm::astore.create_new_pool();
				}
				res = xm::astore.pools.back().second;
				xm::astore.pools.back().second += (numBucket+1)*xma::BUCKET_SIZE;//n+sizeof(size_t);
				*(res-1) = numBucket;
			} else {
				res = xm::astore.buckets[numBucket].back();
				xm::astore.buckets[numBucket].pop_back();
			}
			#ifdef XERUS_PERFORMANCE_ANALYSIS
				xm::astore.allocCount[numBucket] += 1;
				xm::astore.currAlloc[numBucket] += 1;
				if (xm::astore.currAlloc[numBucket] > xm::astore.maxAlloc[numBucket]) {
					xm::astore.maxAlloc[numBucket] = xm::astore.currAlloc[numBucket];
				}
			#endif
			return static_cast<void*>(res);
		}
	}

	void* operator new(std::size_t n) {
		return myalloc(n);
	}

	void *operator new[](std::size_t s) {
		return myalloc(s);
	}

	// void *malloc(size_t size) {
	//     return myalloc(size);
	// }
	// 
	// void *calloc (size_t _nmemb, size_t _size) {
	// 	if (~0ul / _size < _nmemb) {
	// 		throw std::bad_alloc();
	// 	}
	// 	size_t s = _nmemb * _size;
	// 	void *t = myalloc(s);
	// 	memset(t, 0, s);
	// 	return t;
	// }
	// 
	// void *realloc (void *_ptr, size_t _size) {
	// 	size_t n=0;
	// 	if (_ptr != nullptr) {
	// 		n = *(static_cast<size_t*>(_ptr)-2);
	// 	}
	// 	if (n>= _size) {
	// 		return _ptr;
	// 	} else {
	// 		void *newstorage = myalloc(_size);
	// 		memcpy(newstorage, _ptr, n);
	// 		::operator delete(_ptr);
	// 		return newstorage;
	// 	}
	// }
	// TODO valloc


	void mydelete(void *ptr) noexcept {
		uint8_t n = *(static_cast<uint8_t*>(ptr)-1);
		if (n<0xFF) {
			#ifdef XERUS_PERFORMANCE_ANALYSIS
				xm::astore.currAlloc[n] -= 1;
			#endif
			if (programIsRunning) {
				xm::astore.buckets[n].push_back(static_cast<uint8_t*>(ptr));
			}
		} else {
			uint8_t offset = *(static_cast<uint8_t*>(ptr)-2);
			xerus::misc::r_free(static_cast<void*>(static_cast<uint8_t*>(ptr)-xma::ALIGNMENT-1+offset));
		}
	}

	void operator delete(void* ptr) noexcept {
		mydelete(ptr);
	}

	void operator delete[](void* ptr) noexcept {
		mydelete(ptr);
	}
	
	#if __cplusplus >= 201402L
		void operator delete(void* ptr, size_t) noexcept {
			mydelete(ptr);
		}

		void operator delete[](void* ptr, size_t) noexcept {
			mydelete(ptr);
		}
	#endif

#endif
