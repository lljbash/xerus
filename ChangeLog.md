## Changelog

Potentially breaking changes are marked with an exclamation point '!' at the begin of their description.

* 2019-03-?? v4.0.0
 * Added the HTTensor class for Hierarchical Tensor Decompositions with balanced trees.
 * ! Changed the truncated SVD, TensorNetwork::round_edge and TT:round with epsilon to limit their respective relative errors to epsilon.
 * ! Removed support for the replacement allocator.
 * ! Changed some Makefile configs for more flexibility. See the new config.mk.default file and update your own config.
 * ! Major rewrite of MeasurmentSets.
 * ! Support for GCC <= 4.8 was dropped. Update your compiler if this affects you.
 * (optional) support for arpack https://github.com/opencollab/arpack-ng
 * Fixed minor rank reduction bug in TensorNetwork soft thresholding.
 

* 2017-05-31 v3.0.1
 * Added TTNetwork::use_dense_representations() to convert all components to dense representation.
 * Exporting Tensor::dense_copy() and Tensor::sparse_copy() to python.
 * More examples on the homepage and checks for code-coverage.

* 2017-05-29 v3.0.0
 * Python wrapper now stable.
 * ! REQUIRE macro now logs as error instead of fatal error.
 * ! All macros and preprocessor defines now use the XERUS_ prefix. The config.mk file changed accordingly.
 * ! TT::find_largest_entry and TT::dyadic_product left the TT scope.
 * ! Tensor::modify_diag_elements renamed to Tensor::modify_diagonal_entries for naming consistency.
 * Much faster solve of matrix equations Ax=b by exploiting symmetry and definiteness where possible. This directly speeds up the ALS as well.
 * Added Tensor.one_norm() and one_norm(Tensor) to calculate the one norm of a Tensor.
 * New documentation ('make doc') and homepage.
 * Some minor bugfixes and performance improvements.

* 2016-06-23 v2.4.0
 * Introduced nomeclature 'mode'. Marked all functions that will be renamed / removed in v3.0.0 as deprecated.
 * Added an experimental python wrapper when 'OTHER += -DXERUS_EXPERIMENTAL_PYTHON_WRAPPER' is defined in the config file.
 * Added TT::degrees_of_freedom function to determine the dimension of the current tangent space.
 * ! Corrected a typo in the class names of the measurement sets.
 * Added constants xerus::VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION and VERSION_COMMIT.
 * Improved compilation time and more forgiving default settings for warnings.

* 2016-04-04 v2.3.0
 * Added save_to_file and load_from_file functions to store and restore Tensors and TensorNetworks in/from files.
 * Added SVD based calculate_pseudo_inverse function to Tensor.
 * Added implementation for perform_trace() for sparse Tensors.
 * Fixed the pipe operator for xerus::misc::generic_exception to not destroy the type information.
 * Bug fix in the handling of fixed indices in TensorNetworks.
 * Several static member function now warn if their return type is not used.
 * Initial support for compilation with the intel ICC.

* 2016-03-11 v2.2.1
 * Added support for 32bit systems.

* 2016-03-08 v2.2.0
 * Added Sparse solve and sparse QR factorization.
 * ! Xerus now depends on the SuiteSparseQR (-lspqr) library.
 * TTNetwork sum now supports sparse components.
 * TTNetwork entrywise\_product now supports sparse components.
 * ! entrywise\_product (for TTNetworks) now a free function instead of static member.
 * TTNetwork entrywise\_square now supports sparse components.
 * Bugfix in Dense/Sparse conversion factor handling.

* 2016-02-10 v2.1.1
 * Basic support for fixed indices in TensorNetworks.
 * Readded TTNetwork constructor from DimensionTuple.
 * Bugfixes for TTNetwork dyadic_product.

* 2016-02-05 v2.1.0
 * ! fix_slate() now works correctly for TTTensors and (correctly) throws an error for TTOperators.
 * Several bug fixes in the handling of TTNetworks of degree zero.
 * Implemented the Tensor CQ function, which is now used where appropriate.
 * Added several unit tests to ensure consitent results for dense/sparse Tensors, TensorNetwork and TTNetworks.
 * Several Bugfixes in xerus::misc and a bugfix in TT:dyadic_product.

* 2016-01-26 v2.0.0
 * ! Merged FullTensor and SparseTensor into a single Tensor class.
 * Implemented several FullTensor functions also for sparse representations (modify(\_diag)_elements, fix\_slate, resize\_dimension,...)
 * ! IndexedTensor*s now considered internal.
 * ! Removed all support for stored IndexedTensor*s.
 * ! Reworked the IndexedTensor* index assignment.
 * ! Changed resize_dimension to allow slate insertion at the first position.
 * Added TTTensor::random with callback function to manipulate the singular values of all matricisations.
 * Rewrote the ALS algorithm for better readability.
 * Added fully functional (multi-site) DMRG and alternating steepest descent algorithms.
 * Support for low level factorisation calls for Tensor class.
 * Several bug fixes, including SVD factor handling, SparseTensor summation, Tensor resize_dimension, TN evaluation,....
 * Added several test cases.

* 2016-01-14 v1.5.0
 * Added the ADF algorithm for rank one measurements instead of point evaluations.
 * Parallelized the ADF algorithm using openmp (note: at this point not compatible with the replacement allocator!)
 * TT::random no longer uses size_t to measure the size of the tensor and no longer exceeds maximal ranks.
 * Fixed a bug in the ALS that caused the algorithm to always stop after one sweep if called without operator A.

* 2015-10-20 v1.4.2
 * The custom allocator now enforces memory alignment even if the underlying malloc does not.
 * Cannonicalized TT tensors can no longer exceed the maximal TT rank.

* 2015-10-14 v1.4.1
 * Fixed a bug with the recannonicalization of products of TTOperators.
 * Many improvements for the ADF algorithm for tensor completion.
 * Added TTTensor::ones and TTOperator::ones to create TTTensors with all entries equal to 1.

* 2015-09-09 v1.4.0
 * Added an implementation of the ADF algorithm.
 * Added a custom allocator to better handle the small allocations of Indices etc. (requires to set the corresponding option in the config.mk).
 * Reduced overhead (time not spent in blas) of TensorNetwork contractions.
 * Tensor decomposition now allow arbitrary index orders in the target as well as the source.
 * Added TN::measure(...) to perform a batch of evaluations of the network.
 * Added TN::fix_slate(...) to fix a single index to a specific value.
 * Added TN::reduce_representation() to trace out any full-rank edge.
 * Added TN::transfer_core(...) and TN::round_edge(...) analogously to TT rounding / core movement operations.
 * ! renamed class::construct_random to class::random and TTOperator::construct_identity to TTOperator::identity.
 * Added several named constructors:
  * Tensor::ones(...) to construct tensors filled with all entries equal to 1.
  * Tensor::identity(...) to construct a (sparse) tensor equal to the TTOperator::identity.
  * Tensor::dirac(...) to construct a (sparse) tensor with a single 1 at a given position and 0 everywhere else.
  * Tensor::kronecker(...) to construct a tensor that is 1 when all indices are equal and 0 otherwise.
 * ! misc::approx_equal now uses a relative difference (for consistency with the tensor variant xerus::approx_equal).
 * Several bugfixes for SparseTensors.
 * Several minor bugfixes (TN::draw no longer uses temporary files, span-zero indices in assignment, ...).

* 2015-07-10 v1.3.0
 * Added several retractions to allow Riemannian algorithms on TT manifolds.
 * Added the basic (Riemannian) steepest descent algorithm.
 * Added the (so far untested) CG algorithm for TT tensors.
 * ! New PerformanceData class to log stepcount, time and residual of algorithm runs.
 * New algorithm to determine the largest entry of a TT tensor.
 * Fully rewrote the contraction heuristics, now including a local search for more optimal contraction orders.
 * Several bugfixes for order-zero tensors, TT tensors of more than full rank and other rare cases.

* 2015-06-15 v1.2.1
 * entrywise_product for FullTensor and TTNetwork.
 * TensorNetworks now without extraneous factor.

* 2015-06-15 v1.2.0
 * Rank revealing QR decomposition in TTNetwork:move_core. 
 * ALS performance improvement by skipping full-rank nodes.
 * Added TensorNetwork::draw to get a graphical representation of a given tensor network.
 * Added algorithm DecompositionALS to approximate a FullTensor by a TTTensor.
 * New Quick-Start guide.
 * Several bugfixes, more consistent operator availability and much more documentation.

* 2015-06-02 v1.1.1
 * Complete documentation for Tensor, FullTensor and SparseTensor
 * Added implementation of the projection ALS
 * Fixed performance bug in logging system
 * ALS now uses energy functional for convergence testing
 * Several minor fixes

* 2015-06-01 v1.1.0
 * ! New representation for the TT tensors: all component tensors are now of degree 3 / 4 (operators).
 * TT::get_component and ::set_component functions that handle the meta data.
 * More strict ISO c++ conformity: no names starting with '__'.
 * More elaborate performance profiling.

* 2015-05-27 v1.0.2
 * Added support for Clang++
 * ! Cleanup of namespace structure
 * Several documentation fixes

* 2015-05-20 v1.0.1
 * fixed a major bug with the handling of shared resources in TensorNetworks
 * added TTTensor/TTOperator::chop() function
 * some minor fixes

* 2015-05-19 v1.0
 * first stable release
