#include "misc.h"


std::vector<size_t> strides_from_dimensions_and_item_size(const std::vector<size_t>& _dimensions, const size_t _item_size) {
    const size_t ndim = _dimensions.size();
    std::vector<size_t> strides(ndim, 0);
    if (ndim > 0) {
        strides[ndim-1] = _item_size;
        for (size_t i=0; i<ndim-1; ++i) {
            size_t rev_i = ndim-1-i;
            strides[rev_i-1] = _dimensions[rev_i] * strides[rev_i];
        }
    }
    return strides;
}


void expose_tensor(module& m) {
    enum_<Tensor::Representation>(m, "Representation", "Possible representations of Tensor objects.")
        .value("Dense", Tensor::Representation::Dense)
        .value("Sparse", Tensor::Representation::Sparse)
    ;
    enum_<Tensor::Initialisation>(m, "Initialisation", "Possible initialisations of new Tensor objects.")
        .value("Zero", Tensor::Initialisation::Zero)
        .value("None", Tensor::Initialisation::None)
    ;

    class_<Tensor>(m, "Tensor", "a non-decomposed Tensor in either sparse or dense representation", buffer_protocol())
    .def_buffer([](Tensor& t) -> buffer_info {
        return buffer_info(
            t.get_dense_data(),                    /* Pointer to buffer */
            sizeof(value_t),                       /* Size of one scalar */
            format_descriptor<value_t>::format(),  /* Python struct-style format descriptor */
            t.degree(),                            /* Number of dimensions */
            t.dimensions,                          /* Buffer dimensions */
            strides_from_dimensions_and_item_size(t.dimensions, sizeof(value_t))  /* Strides (in bytes) for each index */
        );
    })
    .def(init<>(), "constructs an empty Tensor")
    .def(init<Tensor::DimensionTuple, Tensor::Representation, Tensor::Initialisation>(),
        "constructs a Tensor with the given dimensions",
        arg("dim"),
        arg("repr")=Tensor::Representation::Sparse,
        arg("init")=Tensor::Initialisation::Zero
    )
    .def(init<const TensorNetwork&>())
    .def(init<const Tensor &>())
    /* .def(init<const Tensor::DimensionTuple&, const std::function<value_t(std::vector<size_t>)>>()) */
    .def_static("from_function", +[](const Tensor::DimensionTuple& _dim, const std::function<value_t(std::vector<size_t>)> _f){
        LOG(warning, "Deprecation warning: `from_function` is deprecated and will be removed in Xerus v5.0.0. Use the `Tensor` constructor instead.");
        return Tensor(_dim, _f);
    })
    // .def_static("from_buffer", +[]()
    // .def("from_ndarray", +[](PyObject *_npObject){
    //     //TODO check for dangling pointers!
    //     #pragma GCC diagnostic push
    //     #pragma GCC diagnostic ignored "-Wuseless-cast"
    //     #pragma GCC diagnostic ignored "-Wold-style-cast"
    //     #pragma GCC diagnostic ignored "-Wcast-qual"
    //     #pragma GCC diagnostic ignored "-Wpedantic"
    //     PyArrayObject *npa = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(_npObject, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
    //     #pragma GCC diagnostic pop
    //     int deg = PyArray_NDIM(npa);
    //     std::vector<size_t> dims;
    //     dims.resize(size_t(deg));
    //     if (PyArray_ISCONTIGUOUS(npa)) {
    //  for (int i=0; i<deg; ++i) {
    //      dims[size_t(i)] = size_t(PyArray_DIMS(npa)[i]);
    //  }
    //  Tensor result(dims, Tensor::Representation::Dense, Tensor::Initialisation::None);
    //  misc::copy(result.get_unsanitized_dense_data(), static_cast<double*>(PyArray_DATA(npa)), result.size);
    //  Py_DECREF(npa);
    //  return object(result);
    //     } else if (PyArray_ISFORTRAN(npa)) {
    //  std::vector<size_t> shuffle(dims);
    //  for (int i=0; i<deg; ++i) {
    //      dims[size_t(deg-i-1)] = size_t(PyArray_DIMS(npa)[i]);
    //      shuffle[size_t(deg-i-1)] = size_t(i);
    //  }
    //  Tensor result(dims, Tensor::Representation::Dense, Tensor::Initialisation::None);
    //  // TODO reduce number of copies
    //  Tensor tmp(dims, Tensor::Representation::Dense, Tensor::Initialisation::None);
    //  misc::copy(tmp.get_unsanitized_dense_data(), static_cast<double*>(PyArray_DATA(npa)), result.size);
    //  reshuffle(result, tmp, shuffle);
    //  Py_DECREF(npa);
    //  return object(result);
    //     } else {
    //  LOG(error, "could not convert ndarray of neither c nor fortran striding");
    //  Py_DECREF(npa);
    //  return object();
    //     }
    // }).staticmethod("from_ndarray")
    // .def("to_ndarray", +[](Tensor &_this){
    //     std::vector<npy_intp> dimensions;
    //     for (size_t d : _this.dimensions) {
    //  dimensions.emplace_back(npy_intp(d));
    //     }

    //     #pragma GCC diagnostic push
    //     #pragma GCC diagnostic ignored "-Wuseless-cast"
    //     #pragma GCC diagnostic ignored "-Wold-style-cast"
    //     #pragma GCC diagnostic ignored "-Wcast-qual"
    //     #pragma GCC diagnostic ignored "-Wpedantic"
    //     PyObject *pyObj;
    //     if (_this.is_dense()) {
    //  pyObj = PyArray_SimpleNewFromData(int(_this.degree()), &dimensions[0], NPY_DOUBLE, _this.get_dense_data());
    //     } else {
    //  std::cerr << "RuntimeWarning: converting sparse tensor to dense ndarray" << std::endl;
    //  int nd = int(_this.degree());
    //  npy_intp* dims = &dimensions[0];
    //  int typenum = NPY_DOUBLE;
    //  void* data = calloc(_this.size, sizeof(double));

    //  pyObj = PyArray_New(&PyArray_Type, nd, dims, typenum, NULL, data, 0, NPY_ARRAY_CARRAY | NPY_ARRAY_OWNDATA, NULL);

    //  Tensor tmp(_this); // leaves _this as a sparse tensor
    //  misc::copy(static_cast<double*>(data), tmp.get_dense_data(), tmp.size);
    //     }
    //     //TODO: remove copy?
    //     PyObject * res = PyArray_Copy(reinterpret_cast<PyArrayObject*>(pyObj)); // copy due to lifetime issues (copy is owned by numpy instead of us)
    //     Py_DECREF(pyObj);
    //     return res;
    //     #pragma GCC diagnostic pop
    // })
    .def_property_readonly("dimensions", +[](Tensor &_A) {
        return _A.dimensions;
    })
    .def("degree", &Tensor::degree)
    .def_readonly("factor", &Tensor::factor)
    .def_readonly("size", &Tensor::size)
    .def("one_norm", &Tensor::one_norm)
    .def("frob_norm", &Tensor::frob_norm)
    .def_static("random",
        +[](std::vector<size_t> _dim) {
        return xerus::Tensor::random(_dim);
        },
        "Construct a tensor with i.i.d. Gaussian random entries."
        parametersDocstr
        "dim : list or tuple of int\n"
        "n : list or tuple of int, optional\n"
        "    number of non-zero entries",
        arg("dim")
        )
    .def_static("random",
        +[](std::vector<size_t> _dim, size_t _n) {
        return xerus::Tensor::random(_dim, _n);
        },
        arg("dim"), arg("n")
        )
    .def_static("random_orthogonal",
        +[](std::vector<size_t> _dimLhs, std::vector<size_t> _dimRhs) {
        return xerus::Tensor::random_orthogonal(_dimLhs, _dimRhs);
        })
    .def_static("ones", &Tensor::ones, 
         "Constructs a tensor of given dimensions that is equal to 1 everywhere."
          parametersDocstr "dim : list or tuple of int",
    arg("dim")
    )
    .def_static("identity", &Tensor::identity, 
        "Constructs a Tensor representation of the identity operator with the given dimensions."
        parametersDocstr "dim : list or tuple of int",
arg("dim")
    )
    .def_static("kronecker", &Tensor::kronecker, 
        "Constructs a Tensor representation of the kronecker delta (=1 where all indices are identical, =0 otherwise)."
        parametersDocstr "dim : list or tuple of int",
arg("dim")
    )
    .def_static("dirac", static_cast<Tensor (*)(Tensor::DimensionTuple, const Tensor::MultiIndex&)>(&Tensor::dirac),
        "Construct a Tensor with a single entry equals one and all other zero."
        parametersDocstr
        "dim : list or tuple of int\n"
        "pos : list or tuple of int\n"
        "    position of the 1 entry",
        arg("dim"), arg("pos")
    )
    .def_static("dirac", static_cast<Tensor (*)(Tensor::DimensionTuple, const size_t)>(&Tensor::dirac))
    .def("has_factor", &Tensor::has_factor)
    .def("is_dense", &Tensor::is_dense)
    .def("is_sparse", &Tensor::is_sparse)
    .def("sparsity", &Tensor::sparsity)
    .def("all_entries_valid", &Tensor::all_entries_valid)
    .def("reorder_cost", &Tensor::reorder_cost)
    .def("reinterpret_dimensions", &Tensor::reinterpret_dimensions,
        arg("dim"),
        "Reinterprets the dimensions of the tensor."
        parametersDocstr
        "dim : list or tuple of int"
    )
    .def("resize_mode", &Tensor::resize_mode,
        "Resizes a specific mode of the Tensor."
        parametersDocstr
        "mode : int\n"
        "newDim : int\n"
        "cutPos : int, optional (default: infinity)\n"
        "    The position within the selected mode in front of which slates are inserted or removed.",
        arg("mode"), arg("newDim"), arg("cutPos")=~0ul
    )
    .def("fix_mode", &Tensor::fix_mode,
        "Fixes a specific mode to a specific value, effectively reducing the order by one."
        parametersDocstr
        "mode : int\n"
        "value : int",
        arg("mode"), arg("value")
    )
    .def("remove_slate", &Tensor::remove_slate,
        "Removes a single slate from the Tensor, reducing dimension[mode] by one."
        parametersDocstr
        "mode : int\n"
        "pos : int",
        arg("mode"), arg("pos")
    )
    .def("perform_trace", &Tensor::perform_trace)
    .def("offset_add", &Tensor::offset_add)
    .def("use_dense_representation", &Tensor::use_dense_representation)
    .def("use_sparse_representation", &Tensor::use_sparse_representation,
        arg("epsilon")=EPSILON
    )
    .def("sparse_copy", &Tensor::sparse_copy)
    .def("dense_copy", &Tensor::dense_copy)
    .def("ensure_own_data", &Tensor::ensure_own_data)
    .def("ensure_own_data_no_copy", &Tensor::ensure_own_data_no_copy)
    .def("apply_factor", &Tensor::apply_factor)
    .def("ensure_own_data_and_apply_factor", &Tensor::ensure_own_data_and_apply_factor)
    .def_static("multiIndex_to_position", &Tensor::multiIndex_to_position)
    .def_static("position_to_multiIndex", &Tensor::position_to_multiIndex)
    // .def("__call__", +[](Tensor *_this, const std::vector<Index> &_idx){
    //     return  new xerus::internal::IndexedTensor<Tensor>(std::move((*_this)(_idx)));
    // }, return_value_policy<manage_new_object, with_custodian_and_ward_postcall<0, 1>>() )
    .def("__str__", &Tensor::to_string)
    .def(self * value_t())
    .def(value_t() * self)
    .def(self / value_t())
    .def(self + self)
    .def(self - self)
    .def(self += self)
    .def(self -= self)
    .def("__getitem__", +[](Tensor &_this, size_t _i) {
        if (_i >= _this.size) {
            throw index_error("Index out of range");
        }
        return _this[_i];
    })
    .def("__getitem__", +[](Tensor &_this, std::vector<size_t> _idx) {
        return _this[_idx];
    })
    .def("__setitem__", +[](Tensor &_this, size_t _i, value_t _val) {
        _this[_i] = _val;
    })
    .def("__setitem__", +[](Tensor &_this, std::vector<size_t> _i, value_t _val) {
        _this[_i] = _val;
    })
    ;
}
