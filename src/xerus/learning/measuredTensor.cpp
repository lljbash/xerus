#include "measuredTensor.h"
#include "rankOneMeasuredTensor.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <xerus.h>
#include <xerus/misc/internal.h> // REQUIRE
namespace xe = xerus;
namespace py = pybind11;
using namespace pybind11::literals;
using xerus::misc::operator<<;


namespace pybind11 { namespace detail {
    template <> struct type_caster<xe::Tensor> {
    public:
        /**
         * This macro establishes the name 'xerus.Tensor' in
         * function signatures and declares a local variable
         * 'value' of type xe::Tensor
         */
        PYBIND11_TYPE_CASTER(xe::Tensor, _("xerus.Tensor"));

        /**
         * Conversion part 1 (Python->C++): return false upon failure.
         * The second argument indicates whether implicit conversions should be applied.
         */
        bool load(handle src, bool) {
            /* std::cerr << "Loading Tensor" << std::endl; */
            py::object serialize = py::module::import("xerus").attr("serialize");
            std::string bytes = serialize(src).cast<std::string>();
            /* std::cerr << "Header: " << bytes.substr(0, bytes.find("\n")) << std::endl; */
            /* value = xe::Tensor(xe::misc::deserialize<xe::Tensor>(bytes)); */
            value = xe::misc::deserialize<xe::Tensor>(bytes);
            return true;
        }

        /**
         * Conversion part 2 (C++ -> Python):
         * The second and third arguments are used to indicate the return value policy
         * and parent object (for ``return_value_policy::reference_internal``) and are generally
         * ignored by implicit casters.
         */
        static handle cast(xe::Tensor src, return_value_policy /* policy */, handle /* parent */) {
            /* std::cerr << "Casting Tensor" << std::endl; */
            py::object deserialize = py::module::import("xerus").attr("deserialize");
            return deserialize(py::bytes(xe::misc::serialize(src))).release();
        }
    };

    template <> struct type_caster<xe::TTTensor> {
    public:
        /**
         * This macro establishes the name 'xerus.TTTensor' in
         * function signatures and declares a local variable
         * 'value' of type xe::TTTensor
         */
        PYBIND11_TYPE_CASTER(xe::TTTensor, _("xerus.TTTensor"));

        /**
         * Conversion part 1 (Python->C++): return false upon failure.
         * The second argument indicates whether implicit conversions should be applied.
         */
        bool load(handle src, bool) {
            /* std::cerr << "Loading TTTensor" << std::endl; */
            py::object serialize = py::module::import("xerus").attr("serialize");
            std::string bytes = serialize(src).cast<std::string>();
            /* std::cerr << "Header: " << bytes.substr(0, bytes.find("\n")) << std::endl; */
            /* value = xe::TTTensor(xe::misc::deserialize<xe::TTTensor>(bytes)); */
            value = xe::misc::deserialize<xe::TTTensor>(bytes);
            return true;
        }

        /**
         * Conversion part 2 (C++ -> Python):
         * The second and third arguments are used to indicate the return value policy
         * and parent object (for ``return_value_policy::reference_internal``) and are generally
         * ignored by implicit casters.
         */
        static handle cast(xe::TTTensor src, return_value_policy /* policy */, handle /* parent */) {
            /* std::cerr << "Casting TTTensor" << std::endl; */
            auto deserialize = py::module::import("xerus").attr("deserialize");
            return deserialize(py::bytes(xe::misc::serialize(src))).release();
        }
    };
}} // namespace pybind11::detail


/* template<class T> */
/* T mean(const std::function<T(const size_t)>& fnc, const std::vector<size_t>& batch, const T& init) { */
/*     T ret(init); */

/*     //TODO: schedule, threadprivate(dyadComp, tmp) */
/*     #pragma omp parallel for reduction(+: ret) firstprivate(fnc, batch) default(none) */
/*     for(size_t i=0; i<batch.size(); i++) { */
/*         ret += fnc(batch[i]); */
/*     } */

/*     return ret / batch.size(); */
/* } */


/* template<> */
double mean(const std::function<double(const size_t)>& fnc, const std::vector<size_t>& batch, const double& init) {
    double ret(init);

    //TODO: schedule, threadprivate(dyadComp, tmp)
    #pragma omp parallel for reduction(+: ret) firstprivate(fnc, batch) default(none)
    for(size_t i=0; i<batch.size(); i++) {
        ret += fnc(batch[i]);
    }

    return ret / batch.size();
}


/* template<> */
xe::Tensor mean(const std::function<xe::Tensor(const size_t)>& fnc, const std::vector<size_t>& batch, const xe::Tensor& init) {
    xe::Tensor ret(init);

    //TODO: schedule, threadprivate(dyadComp, tmp)
    #pragma omp declare reduction(+: xe::Tensor: omp_out += omp_in) initializer(omp_priv = xe::Tensor(omp_orig.dimensions))
    #pragma omp parallel for reduction(+: ret) firstprivate(fnc, batch) default(none)
    for(size_t i=0; i<batch.size(); i++) {
        ret += fnc(batch[i]);
    }

    return ret / batch.size();
}


size_t argmax(const xe::Tensor& x) {
    const double* x_arr = x.get_unsanitized_dense_data();
    return std::max_element(x_arr, x_arr+x.size) - x_arr;
}


double local_error(const xe::Tensor& x, const size_t y) {
    return double(argmax(x) != y);
}


double error(const xe::RankOneMeasuredTTTensor& meas, const std::vector<size_t>& ref, const std::vector<size_t>& batch) {
    REQUIRE(meas.value_dimensions().size() == 1, "`meas.value` must be a vector");
    double ret = 0;
    size_t j;

    //TODO: schedule, threadprivate(dyadComp, tmp)
    #pragma omp parallel for reduction(+: ret) private(j) shared(meas, ref, batch) default(none)
    for(size_t i=0; i<batch.size(); i++) {
        j = batch[i];
        ret += local_error(meas.value(j), ref[j]);
    }

    return ret / batch.size();
}


double LSE(const xe::Tensor& x) {
    const double factor = x.factor;
    const double* x_arr = x.get_unsanitized_dense_data();
    const double x_max = *std::max_element(x_arr, x_arr+x.size);  // <algorithm>
    double ret = 0;
    for (size_t i=0; i<x.size; i++) { ret += exp(factor*(x_arr[i]-x_max)); }
    return log(ret) + x_max;
    /* return log(std::transform_reduce(std::execution::par_unseq, x.begin(), x.end(), 0.0, std::plus<>(), [](const double x){ return exp(x-x_max); }));  // <numeric>, <execution> */
}


double local_loss(const xe::Tensor& x, const size_t y) {
    return LSE(x) - x[y];
}


double loss(const xe::RankOneMeasuredTTTensor& meas, const std::vector<size_t>& ref, const std::vector<size_t>& batch) {
    REQUIRE(meas.value_dimensions().size() == 1, "`meas.value` must be a vector");
    double ret = 0;
    size_t j;

    //TODO: schedule, threadprivate(dyadComp, tmp)
    #pragma omp parallel for reduction(+: ret) private(j) shared(meas, ref, batch) default(none)
    for(size_t i=0; i<batch.size(); i++) {
        j = batch[i];
        ret += local_loss(meas.value(j), ref[j]);
    }

    return ret / batch.size();
}


xe::Tensor softmax(const xe::Tensor& x) {
    REQUIRE(x.dimensions.size() == 1, "`x` must be a vector");
    xe::Tensor ret(x);
    double* ret_arr = ret.get_dense_data();  // ensures onw data and applies factor
    const double ret_max = *std::max_element(ret_arr, ret_arr+ret.size);  // <algorithm>
    double ret_sum = 0;
    for (size_t i=0; i<x.size; i++) {
        ret_arr[i] = exp(ret_arr[i] - ret_max);
        ret_sum += ret_arr[i];
    }
    return ret / ret_sum;
}


xe::Tensor local_d_loss(const xe::Tensor& x, const xe::TensorNetwork& dx, const size_t y) {
    xe::Tensor ret;
    const xe::Index m,a;  // measured, accumulated
    const auto s = softmax(x);
    const auto v = xe::Tensor::dirac(s.dimensions, y);  // reuse a global tensor --> cheaper
    ret(m&0) = (s-v)(a) * dx(a,m&1);
    return ret;
}


xe::Tensor d_loss(const xe::RankOneMeasuredTTTensor& meas, const std::vector<size_t>& ref, const std::vector<size_t>& batch) {
    const auto& val_dims = meas.value_dimensions();
    REQUIRE(val_dims.size() == 1, "`meas.value` must be a vector");
    const auto& core_dims = meas.get_core().dimensions;
    xe::Tensor ret(core_dims);
    size_t j;

    //TODO: schedule, threadprivate(dyadComp, tmp)
    #pragma omp declare reduction(+: xe::Tensor: omp_out += omp_in) initializer(omp_priv = xe::Tensor(omp_orig.dimensions))
    #pragma omp parallel for reduction(+: ret) private(j) shared(meas, ref, batch) default(none)
    for(size_t i=0; i<batch.size(); i++) {
        j = batch[i];
        ret += local_d_loss(meas.value(j), meas.gradient(j), ref[j]);
    }

    return ret / batch.size();
}



PYBIND11_MODULE(measuredTensor, m) {
    m.doc() = "pybind11 bindings for `measuredTensor`"; // module docstring

    //TODO: docstrings
    py::class_<xe::RankOneMeasuredTTTensor>(m, "RankOneMeasuredTTTensor")
        .def(py::init<>())
        .def(py::init<xe::TTTensor&, std::vector<std::vector<xe::Tensor>>&>()) //, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("numComponents", &xe::RankOneMeasuredTTTensor::get_numComponents)
        .def_property_readonly("numMeasurements", &xe::RankOneMeasuredTTTensor::get_numMeasurements)
        .def_property("corePosition", &xe::RankOneMeasuredTTTensor::get_corePosition, &xe::RankOneMeasuredTTTensor::set_corePosition) //, py::call_guard<py::gil_scoped_release>())
        .def_property("core", &xe::RankOneMeasuredTTTensor::get_core, &xe::RankOneMeasuredTTTensor::set_core)
        .def("value", &xe::RankOneMeasuredTTTensor::value, "The value of the measurement at the given index", "index"_a)
        /* .def("gradient") */
        .def_property_readonly("measuredTensor", &xe::RankOneMeasuredTTTensor::get_tensor, py::return_value_policy::take_ownership);

    /* /1* m.def("mean", [](const std::function<Tensor(const size_t)>& fnc, const std::vector<size_t>& batch){ return xe::mean(fnc, batch); }, "A function which adds two numbers", "function"_a, "batch"_a, "init"_a=); *1/ */
    /* m.def("mean", static_cast<double(*)(const std::function<double(const size_t)>&, const std::vector<size_t>&, const double&)>(&xe::mean), "Compute the mean of `function` applied to `batch`", "function"_a, "batch"_a, "init"_a); //, py::call_guard<py::gil_scoped_release>()); */
    /* m.def("mean", static_cast<xe::Tensor(*)(const std::function<xe::Tensor(const size_t)>&, const std::vector<size_t>&, const xe::Tensor&)>(&xe::mean), "Compute the mean of `function` applied to `batch`", "function"_a, "batch"_a, "init"_a); //, py::call_guard<py::gil_scoped_release>()); */
    /* /1* m.def("mean", &xe::mean, "Compute the mean of `function` applied to `batch`", "function"_a, "batch"_a, "init"_a, py::call_guard<py::gil_scoped_release>()); *1/ */
    /* m.def("mean_value", [](const xe::RankOneMeasuredTTTensor& mtt, const std::vector<size_t>& batch) { */
    /*           xe::Tensor init(mtt.value_dimensions()); */
    /*           return xe::mean([&](const size_t idx){ return mtt.value(idx); }, batch, init); */
    /*       }, */
    /*       "Compute the mean of `measured_tt.value` applied to `batch`", "measured_tt"_a, "batch"_a); //, py::call_guard<py::gil_scoped_release>()); */

    /* m.def("error", &error, py::arg("measured_tt"), py::arg("reference_values").noconvert(), py::arg("batch").noconvert(), py::call_guard<py::gil_scoped_release>()); */
    m.def("error_fast", &error); //, py::call_guard<py::gil_scoped_release>());
    m.def("loss_fast", &loss); //, py::call_guard<py::gil_scoped_release>());
    m.def("d_loss_fast", &d_loss); //, py::call_guard<py::gil_scoped_release>());

    m.def("error", [&](const xe::RankOneMeasuredTTTensor& meas, const std::vector<size_t>& ref, const std::vector<size_t>& batch) -> double {
        /* return mean<double>([&](const size_t idx){ return local_error(meas.value(idx), ref[idx]); }, batch, 0); */
        return mean([&](const size_t idx){ return local_error(meas.value(idx), ref[idx]); }, batch, 0);
    });
    m.def("error", [&](const xe::RankOneMeasuredTTTensor& meas, const std::vector<size_t>& ref, const std::vector<size_t>& batch, const xe::RankOneMeasuredTTTensor& offset) -> double {
        /* return mean<double>([&](const size_t idx){ return local_error(meas.value(idx)+offset.value(idx), ref[idx]); }, batch, 0); */
        return mean([&](const size_t idx){ return local_error(meas.value(idx)+offset.value(idx), ref[idx]); }, batch, 0);
    });
    m.def("loss", [&](const xe::RankOneMeasuredTTTensor& meas, const std::vector<size_t>& ref, const std::vector<size_t>& batch) -> double {
        return mean([&](const size_t idx){ return local_loss(meas.value(idx), ref[idx]); }, batch, 0);
    });
    m.def("loss", [&](const xe::RankOneMeasuredTTTensor& meas, const std::vector<size_t>& ref, const std::vector<size_t>& batch, const xe::RankOneMeasuredTTTensor& offset) -> double {
        return mean([&](const size_t idx){ return local_loss(meas.value(idx)+offset.value(idx), ref[idx]); }, batch, 0);
    });
    m.def("d_loss", [&](const xe::RankOneMeasuredTTTensor& meas, const std::vector<size_t>& ref, const std::vector<size_t>& batch) -> xe::Tensor {
        xe::Tensor init(meas.get_core().dimensions);
        return mean([&](const size_t idx){ return local_d_loss(meas.value(idx), meas.gradient(idx), ref[idx]); }, batch, init);
    });
    m.def("d_loss", [&](const xe::RankOneMeasuredTTTensor& meas, const std::vector<size_t>& ref, const std::vector<size_t>& batch, const xe::RankOneMeasuredTTTensor& offset) -> xe::Tensor {
        xe::Tensor init(meas.get_core().dimensions);
        return mean([&](const size_t idx){ return local_d_loss(meas.value(idx)+offset.value(idx), meas.gradient(idx), ref[idx]); }, batch, init);
    });

}
