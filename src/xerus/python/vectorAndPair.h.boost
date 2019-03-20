
template<typename containedType>
struct custom_vector_to_list{
	static PyObject* convert(const std::vector<containedType>& v){
		list ret;
		for(const containedType& e : v) {
			ret.append(e);
		}
		return incref(ret.ptr());
	}
};
template<typename containedType>
struct custom_vector_from_seq{
	custom_vector_from_seq(){ converter::registry::push_back(&convertible,&construct,type_id<std::vector<containedType> >()); }
	static void* convertible(PyObject* obj_ptr){
		// the second condition is important, for some reason otherwise there were attempted conversions of Body to list which failed afterwards.
		if(!PySequence_Check(obj_ptr) || !PyObject_HasAttrString(obj_ptr,"__len__")) return 0;
		return obj_ptr;
	}
	static void construct(PyObject* obj_ptr, converter::rvalue_from_python_stage1_data* data){
		void* storage=(reinterpret_cast<converter::rvalue_from_python_storage<std::vector<containedType> >*>(data))->storage.bytes;
		new (storage) std::vector<containedType>();
		std::vector<containedType>* v=reinterpret_cast<std::vector<containedType>*>(storage);
		Py_ssize_t l=PySequence_Size(obj_ptr); if(l<0) abort();
		v->reserve(size_t(l)); 
		for (Py_ssize_t i=0; i<l; i++) { 
			v->push_back(extract<containedType>(PySequence_GetItem(obj_ptr,i))); 
		}
		data->convertible=storage;
	}
};

/* Expose pairs */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"

namespace py = boost::python;

template<typename T1, typename T2>
struct PairToPythonConverter {
	static PyObject* convert(const std::pair<T1, T2>& pair)
	{
		return py::incref(py::make_tuple(pair.first, pair.second).ptr());
	}
};

template<typename T1, typename T2>
struct PythonToPairConverter {
	PythonToPairConverter()
	{
		py::converter::registry::push_back(&convertible, &construct, py::type_id<std::pair<T1, T2> >());
	}
	static void* convertible(PyObject* obj)
	{
		if (!PyTuple_CheckExact(obj)) return 0;
		if (PyTuple_Size(obj) != 2) return 0;
		return obj;
	}
	static void construct(PyObject* obj, py::converter::rvalue_from_python_stage1_data* data)
	{
		py::tuple tuple(py::borrowed(obj));
		void* storage = ((py::converter::rvalue_from_python_storage<std::pair<T1, T2> >*) data)->storage.bytes;
		new (storage) std::pair<T1, T2>(py::extract<T1>(tuple[0]), py::extract<T2>(tuple[1]));
		data->convertible = storage;
	}
	
};

template<typename T1, typename T2>
struct py_pair {
	py::to_python_converter<std::pair<T1, T2>, PairToPythonConverter<T1, T2> > toPy;
	PythonToPairConverter<T1, T2> fromPy;
};


#pragma GCC diagnostic pop
/* End expose pair */
