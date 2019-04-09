#include "rankOneMeasuredTensor.h"
#include <xerus/misc/internal.h> // REQUIRE


namespace xerus {
    template<class dtype>
    std::vector<std::vector<dtype>> transpose(std::vector<std::vector<dtype>>& _vec) {
        const size_t n=_vec.size();
        REQUIRE(n > 0, "Number of rows of `_vec` must be positive");
        const size_t m=_vec[0].size();
        REQUIRE(m > 0, "Number of columns of `_vec` must be positive");
        for (size_t j=0; j<m; j++) {
            REQUIRE(_vec[j].size() == m, "`_vec` must be homogeneous");
        }
        std::vector<std::vector<Tensor>> ret;
        ret.reserve(m);
        for (size_t j=0; j<m; j++) {
            std::vector<dtype> row;
            row.reserve(n);
            for (size_t i=0; i<n; i++) {
                row.push_back(_vec[i][j]);
            }
            ret.push_back(row);
        }
        return ret;
    }

    RankOneMeasuredTTTensor::RankOneMeasuredTTTensor(TTTensor& _tensor, std::vector<std::vector<Tensor>>& _rk1_measurement_set)
        : numComponents(_tensor.degree()),
          numMeasurements(_rk1_measurement_set.size()),
          left_stack_size(0),
          right_stack_size(0),
          tensor(_tensor),
          rk1_measurement_set(transpose(_rk1_measurement_set)),
          left_stack(numComponents, std::vector<Tensor>(numMeasurements)),
          right_stack(numComponents, std::vector<Tensor>(numMeasurements)) {
              REQUIRE(rk1_measurement_set.size() >= numComponents, "measurement set has not enought component measurements for given tensor: " <<  rk1_measurement_set.size() << " vs " << numComponents);
              /* tensor.move_core(0); */
              if (tensor.corePosition != 0) {
                  std::cerr << "WARNING: measuredTensor.corePosition = " << tensor.corePosition << std::endl;
              }
              auto ones = Tensor::ones({1});
              for (size_t i=0; i<numMeasurements; i++) {
                  left_stack[0][i] = ones;
                  right_stack[0][i] = ones;
              }
              left_stack_size = 1;
              right_stack_size = 1;
              //TODO: more tests
              update_left_stack();
              update_right_stack();
    }

    Tensor RankOneMeasuredTTTensor::value(const size_t measIdx) const {
        const Index m,a;  // measured, accumulated

        const auto& core = get_core();
        const auto md = core.degree();  // measured degree
        const auto grad = gradient(measIdx);

        Tensor val;
        val(a&0) = grad(a&md,m^md) * core(m&0);
        return val;
    }

    TensorNetwork RankOneMeasuredTTTensor::gradient(const size_t measIdx) const {
        const Index la,l,e,r,n,ra;  // left accumulated, left, external, right, new, right accumulated

        const auto left = left_stack[left_stack_size-1][measIdx];
        const auto lad = left.degree()-1;   // left accumulated degree
        const auto core = get_core();
        const auto ed = core.degree()-2;    // external degree
        const auto right = right_stack[right_stack_size-1][measIdx];
        const auto rad = right.degree()-1;  // right accumulated degree
        const auto meas = rk1_measurement_set[get_corePosition()][measIdx];
        const auto nd = meas.degree()-ed;

        TensorNetwork grad;
        grad(la^lad,n^nd,ra^rad,l,e^ed,r) = left(la^lad,l) * meas(e^ed,n^nd) * right(r,ra^rad);
        return grad;
    }

    void RankOneMeasuredTTTensor::update_left_stack() {
        const auto corePosition = get_corePosition();
        // `left_stack[pos]` stores the contractions of all components
        // from `0` (incl.) to `pos-1` (incl.) and their respective measurements.
        // Therefore, `left_stack` can have at most `numComponents` many entries
        // and the first entry (`left_stack[0]`) is alway full of `Tensor::ones`.
        // The current top of `left_stack` is located at `left_stack_size-1`.
        //
        // `left_stack` is used to calculate the measurements of `tensor`
        // by contracting the tops of `left_stack` and `right_stack`,
        // the core of `tensor` and its respective measurement.
        // For this `left_stack` has to be consistent with the `corePosition`
        // i.e. `left_stack_size-1 == corePosition`.
        while (left_stack_size > corePosition+1) {
            // core moved left
            /* std::cout << "Left stack: reset old top" << std::endl; */
            auto& old_top = left_stack[left_stack_size-1];
            #pragma omp parallel for default(none) shared(old_top)
            for (size_t measIdx=0; measIdx<numMeasurements; measIdx++) {
                old_top[measIdx].reset();
            }
            left_stack_size--;
        }
        const Index a,l,e,r,n;  // accumulated, left, external, right, new
        while (left_stack_size < corePosition+1) {
            // core moved right
            /* std::cout << "Left stack: measure new top" << std::endl; */
            const auto& old_top = left_stack[left_stack_size-1];
            const auto& cmp = tensor.get_component(left_stack_size-1);
            const auto& meas = rk1_measurement_set[left_stack_size-1];
            const size_t nd = meas[0].degree()-(cmp.degree()-2);  // new degree (assumed constant for all `measIdx`)
            auto& new_top = left_stack[left_stack_size];
            //TODO: `firstprivate` of references might lead to segfaults
            #pragma omp parallel for default(none) firstprivate(cmp) shared(old_top,meas,new_top)
            for (size_t measIdx=0; measIdx<numMeasurements; measIdx++) {
                // The dimensions vector of every measurement for component `cmpIdx`
                // must decompose as (`external` + `new`) where `external` contains
                // all external dimensions of `tensor.get_component(cmpIdx)`.
                const Index a,l,e,r,n;  // accumulated, left, external, right, new
                new_top[measIdx](a&(nd+1),n^nd,r) = old_top[measIdx](a&1,l)  * cmp(l,e&2,r) * meas[measIdx](e&nd,n^nd);
            }
            left_stack_size++;
        }
    }

    void RankOneMeasuredTTTensor::update_right_stack() {
        const auto corePosition = get_corePosition();
        // `mirror(pos)` takes a component position `pos` in `tensor` and returns the position as counted from the right.
        const auto mirror = [&](const size_t position) -> size_t {
            return numComponents-1-position;
        };
        // `right_stack[pos]` stores the contractions of all components
        // from `mirror(pos)+1` (incl.) to `numComponents-1` (incl.) and their respective measurements.
        // Therefore, `right_stack` can have at most `numComponents` many entries
        // and the first entry (`right_stack[0]`) is alway full of `Tensor::ones`.
        // The current top of `right_stack` is located at `right_stack_size-1`.
        //
        // `right_stack` is used to calculate the measurements of `tensor`
        // by contracting the tops of `left_stack` and `right_stack`,
        // the core of `tensor` and its respective measurement.
        // For this `right_stack` has to be consistent with the `corePosition`
        // i.e. `right_stack_size-1 == mirror(corePosition)`.
        while (right_stack_size > mirror(corePosition)+1) {
            // core moved right
            /* std::cout << "Right stack: reset old top" << std::endl; */
            auto& old_top = right_stack[right_stack_size-1];
            #pragma omp parallel for default(none) shared(old_top)
            for (size_t measIdx=0; measIdx<numMeasurements; measIdx++) {
                old_top[measIdx].reset();
            }
            right_stack_size--;
        }
        while (right_stack_size < mirror(corePosition)+1) {
            // core moved right
            /* std::cout << "Right stack: measure new top" << std::endl; */
            const auto& old_top = right_stack[right_stack_size-1];
            const auto& cmp = tensor.get_component(mirror(right_stack_size-1));
            const auto& meas = rk1_measurement_set[mirror(right_stack_size-1)];
            const size_t nd = meas[0].degree()-(cmp.degree()-2);  // new degree (assumed constant for all `measIdx`)
            auto& new_top = right_stack[right_stack_size];
            //TODO: `firstprivate` of references might lead to segfaults
            #pragma omp parallel for default(none) firstprivate(cmp) shared(old_top,meas,new_top)
            for (size_t measIdx=0; measIdx<numMeasurements; measIdx++) {
                // The dimensions vector of every measurement for component `cmpIdx`
                // must decompose as (`external` + `new`) where `external` contains
                // all external dimensions of `tensor.get_component(cmpIdx)`.
                //TODO: test this!!!
                const Index a,l,e,r,n;  // accumulated, left, external, right, new
                new_top[measIdx](l,n^nd,a&(nd+1)) = meas[measIdx](e&nd,n^nd) * cmp(l,e&2,r) * old_top[measIdx](r,a&1);
            }
            right_stack_size++;
        }
    }

    std::vector<size_t> RankOneMeasuredTTTensor::value_dimensions() const {
        std::vector<size_t> dims;
        for (size_t cmpIdx=0; cmpIdx < numComponents; cmpIdx++) {
            /* for (const auto dim : rk1_measurement_set[cmpIdx][0].dimensions) */
            const auto& cmpDims = tensor.get_component(cmpIdx).dimensions;
            const auto& measDims = rk1_measurement_set[cmpIdx][0].dimensions;
            //TODO: test this in constructor!!!
            REQUIRE(cmpDims.size() <= measDims.size()+2, "IE");
            for (size_t dimIdx=1; dimIdx < cmpDims.size()-1; dimIdx++) {
                REQUIRE(cmpDims[dimIdx] == measDims[dimIdx-1], "IE");
            }
            for (size_t dimIdx=cmpDims.size()-2; dimIdx < measDims.size(); dimIdx++) {
                dims.push_back(measDims[dimIdx]);
            }
        }
        return dims;
    }

    std::vector<size_t> RankOneMeasuredTTTensor::gradient_dimensions() const {
        std::vector<size_t> dims(value_dimensions());
        const auto& core_dims = get_core().dimensions;
        dims.insert(dims.end(), core_dims.begin(), core_dims.end());
        return dims;
    }

    /* double mean(const std::function<double(const size_t)>& fnc, const std::vector<size_t>& batch, const double& init) { */
    /*     std::cerr << "double_mean" << std::endl; */
    /*     double ret = init; */

    /*     //TODO: schedule, threadprivate(dyadComp, tmp) */
    /*     #pragma omp parallel for reduction(+: ret) firstprivate(fnc, batch) default(none) */
    /*     for(size_t i=0; i<batch.size(); i++) { */
    /*         ret += fnc(batch[i]); */
    /*     } */

    /*     return ret / batch.size(); */
    /* } */
    /* Tensor mean(const std::function<Tensor(const size_t)>& fnc, const std::vector<size_t>& batch, const Tensor& init) { */
    /*     std::cerr << "Tensor_mean" << std::endl; */
    /*     Tensor ret = init; */

    /*     //TODO: schedule, threadprivate(dyadComp, tmp) */
    /*     #pragma omp declare reduction(+: Tensor: omp_out += omp_in) initializer(omp_priv = Tensor(omp_orig.dimensions)) */
    /*     #pragma omp parallel for reduction(+: ret) firstprivate(fnc, batch) default(none) */
    /*     for(size_t i=0; i<batch.size(); i++) { */
    /*         ret += fnc(batch[i]); */
    /*     } */

    /*     return ret / batch.size(); */
    /* } */
}
