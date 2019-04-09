#include <chrono>
#include <xerus.h>
#include <xerus/misc/internal.h> // REQUIRE
#include "rankOneMeasuredTensor.h"
/* #include <numeric> */
/* #include <execution> */


size_t argmax(const xerus::Tensor& x) {
    const double* x_arr = x.get_unsanitized_dense_data();
    return std::max_element(x_arr, x_arr+x.size) - x_arr;
}

double error(const xerus::RankOneMeasuredTTTensor& meas, const std::vector<size_t>& ref, const std::vector<size_t>& batch) {
    double ret = 0;

    //TODO: schedule, threadprivate(dyadComp, tmp)
    #pragma omp parallel for reduction(+: ret) shared(meas, ref, batch) default(none)
    for(size_t i=0; i<batch.size(); i++) {
        ret += (argmax(meas.value(batch[i])) != ref[batch[i]]);
    }

    return ret / batch.size();
}

/* def LSE(x): */
/*     x_max = np.max(x) */
/*     return np.log(np.sum(np.exp(x-x_max))) + x_max */

double LSE(const xerus::Tensor& x) {
    const double factor = x.factor;
    const double* x_arr = x.get_unsanitized_dense_data();
    const double x_max = *std::max_element(x_arr, x_arr+x.size);  // <algorithm>
    double ret = 0;
    for (size_t i=0; i<x.size; i++) { ret += exp(factor*(x_arr[i]-x_max)); }
    return log(ret) + x_max;
    /* return log(std::transform_reduce(std::execution::par_unseq, x.begin(), x.end(), 0.0, std::plus<>(), [](const double x){ return exp(x-x_max); }));  // <numeric>, <execution> */
}

/* def loss(idx): */
/*     x = meas.value(idx).to_ndarray() */
/*     return LSE(x) - x[ys[idx]] */

double loss(const xerus::RankOneMeasuredTTTensor& meas, const std::vector<size_t>& ref, const std::vector<size_t>& batch) {
    double ret = 0;

    //TODO: schedule, threadprivate(dyadComp, tmp)
    #pragma omp parallel for reduction(+: ret) shared(meas, ref, batch) default(none)
    for(size_t i=0; i<batch.size(); i++) {
        const xerus::Tensor v = meas.value(batch[i]);
        ret += (LSE(v) - v[ref[batch[i]]]);
    }

    return ret / batch.size();
}

/* def softmax(x): */
/*     ret = x.to_ndarray() */
/*     ret -= np.max(ret) */
/*     assert ret.ndim == 1 */
/*     ret = np.exp(ret) */
/*     ret /= np.sum(ret) */
/*     ret = xe.Tensor.from_ndarray(ret) */
/*     return ret */

xerus::Tensor softmax(const xerus::Tensor& x) {
    REQUIRE(x.dimensions.size() == 1, "`x` must be a vector");
    xerus::Tensor ret(x);
    double* ret_arr = ret.get_dense_data();  // ensures onw data and applies factor
    const double ret_max = *std::max_element(ret_arr, ret_arr+ret.size);  // <algorithm>
    double ret_sum = 0;
    for (size_t i=0; i<x.size; i++) {
        ret_arr[i] = exp(ret_arr[i] - ret_max);
        ret_sum += ret_arr[i];
    }
    return ret / ret_sum;
}

/* def d_loss(idx): */
/*     m,a = xe.indices(2)  # measured, accumulated */
/*     s = softmax(meas.value(idx)) */
/*     v = xe.Tensor.dirac([10], int(ys[idx])) */
/*     ret = xe.Tensor() */
/*     ret(m&0) << (s-v)(a) * dx(a,m&1) */
/*     return ret */

xerus::Tensor d_loss(const xerus::RankOneMeasuredTTTensor& meas, const std::vector<size_t>& ref, const std::vector<size_t>& batch) {
    const auto& val_dims = meas.value_dimensions();
    REQUIRE(val_dims.size() == 1, "`meas.value` must be a vector");
    const xerus::Index m,a;  // measured, accumulated
    const auto& core_dims = meas.get_core().dimensions;
    xerus::Tensor ret(core_dims);
    xerus::Tensor tmp(core_dims);

    //TODO: schedule, threadprivate(dyadComp, tmp)
    #pragma omp declare reduction(+: xerus::Tensor: omp_out += omp_in) initializer(omp_priv = xerus::Tensor(omp_orig.dimensions))
    #pragma omp parallel for reduction(+: ret) firstprivate(val_dims, tmp) shared(meas, ref, batch) default(none)
    for(size_t i=0; i<batch.size(); i++) {
        const auto s = softmax(meas.value(batch[i]));
        const auto v = xerus::Tensor::dirac(val_dims, ref[batch[i]]);  // reuse a global tensor --> cheaper
        const auto dx = meas.gradient(batch[i]);
        tmp(m&0) = (s-v)(a) * dx(a,m&1);
        ret += tmp;
    }

    return ret / batch.size();
}



int main() {
    const size_t order = 10,
                 dimension = 10,
                 rank = 20,
                 numMeasurements = 100000;
    const std::vector<size_t> first_cmp_meas = {10, 10};  // {10};

    xerus::TTTensor result;
    result = xerus::TTTensor::random(std::vector<size_t>(order, dimension),
                                     std::vector<size_t>(order-1, rank));
    std::string bytes = xerus::misc::serialize(result);  //TODO: check `deserialize` with rvalue
    REQUIRE(xerus::frob_norm(result-xerus::misc::deserialize<xerus::TTTensor>(bytes))/xerus::frob_norm(result) < 1e-14, "IE");

    std::cout << "Generating " << numMeasurements <<" random measures..." << std::endl;
    std::vector<std::vector<xerus::Tensor>> rk1ms;
    std::vector<size_t> refs;
    rk1ms.reserve(numMeasurements);
    refs.reserve(numMeasurements);
    for (size_t i=0; i<numMeasurements; i++) {
        std::vector<xerus::Tensor> rk1m;
        rk1m.reserve(order);
        rk1m.push_back(xerus::Tensor::random(first_cmp_meas));
        for (size_t j=1; j<order; j++) {
            rk1m.push_back(xerus::Tensor::random(std::vector<size_t>(1, dimension)));
        }
        rk1ms.push_back(rk1m);
        /* refs.push_back(std::rand() % 10);  // biased */
        refs.push_back(std::rand()/((RAND_MAX + 1u)/10));
    }

    //TODO: check if the generic mean function is slower than a more specific mean method of RankOneMeasuredTTTensor
    std::cout << "Initializing MeasuredTensor..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    xerus::RankOneMeasuredTTTensor test(result, rk1ms);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed time: " << elapsed.count() << "s" << std::endl;

    std::cout << "numComponents: " << test.get_numComponents() << std::endl;
    std::cout << "numMeasurements: " << test.get_numMeasurements() << std::endl;

    std::cout << "corePosition: " << test.get_corePosition() << std::endl;
    std::cout << "Moving core..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    test.set_corePosition(1);
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << "Elapsed time: " << elapsed.count() << "s" << std::endl;
    std::cout << "corePosition: " << test.get_corePosition() << std::endl;
    bytes = xerus::misc::serialize(test.get_tensor());
    REQUIRE(xerus::frob_norm(result-xerus::misc::deserialize<xerus::TTTensor>(bytes))/xerus::frob_norm(result) < 1e-14, "IE");

    std::cout << "Computing mean..." << std::endl;
    std::cout << "Batch size: " << test.get_numMeasurements() << std::endl;
    std::vector<size_t> batch;
    batch.reserve(test.get_numMeasurements());
    for (size_t i=0; i<test.get_numMeasurements(); i++) {
        batch.push_back(i);
    }
    xerus::Tensor mean = xerus::Tensor({10});
    start = std::chrono::high_resolution_clock::now();
    mean = xerus::mean([&](const size_t idx){ return test.value(idx); }, batch, mean);
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << "Elapsed time: " << elapsed.count() << "s" << std::endl;
    std::cout << "Mean value: " << mean << std::endl;

    start = std::chrono::high_resolution_clock::now();
    const auto err = error(test, refs, batch);
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << "Elapsed time: " << elapsed.count() << "s" << std::endl;
    std::cout << "Classification Error: " << 100*err << "%" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    const auto los = loss(test, refs, batch);
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << "Elapsed time: " << elapsed.count() << "s" << std::endl;
    std::cout << "Loss: " << los << std::endl;

    start = std::chrono::high_resolution_clock::now();
    const auto dlos = d_loss(test, refs, batch);
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << "Elapsed time: " << elapsed.count() << "s" << std::endl;
    std::cout << "D(Loss): " << los << std::endl;

    std::cout << "Exiting" << std::endl;
    return 0;
}
