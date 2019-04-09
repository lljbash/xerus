#pragma once

#include <xerus.h>
#include "measuredTensor.h"


namespace xerus {
    class RankOneMeasuredTTTensor : public MeasuredTensor<TTTensor> {
    private:
        const size_t numComponents;    //TODO: remove!
        const size_t numMeasurements;  //TODO: remove!
        size_t left_stack_size;
        size_t right_stack_size;
        TTTensor tensor;
        const std::vector<std::vector<Tensor>> rk1_measurement_set;  //TODO: reference?
        std::vector<std::vector<Tensor>> left_stack;
        std::vector<std::vector<Tensor>> right_stack;
    public:
        RankOneMeasuredTTTensor() : numComponents(0), numMeasurements(0), left_stack_size(0), right_stack_size(0) {};
        RankOneMeasuredTTTensor(TTTensor& _tensor, std::vector<std::vector<Tensor>>& _rk1_measurement_set);
        size_t get_numMeasurements() const { return numMeasurements; };
        size_t get_numComponents() const { return numComponents; };
        Tensor value(const size_t measIdx) const;
        TensorNetwork gradient(const size_t measIdx) const;
        const TTTensor& get_tensor() const { return tensor; };
        const Tensor& get_core() const { return tensor.get_component(get_corePosition()); };
        void set_core(const Tensor& val) { tensor.set_component(get_corePosition(), val); };
        size_t get_corePosition() const { return tensor.corePosition; };
        void set_corePosition(const size_t pos) {
            tensor.move_core(pos);
            update_left_stack();
            update_right_stack();
        };
        std::vector<size_t> value_dimensions() const;
        std::vector<size_t> gradient_dimensions() const;
    private:
        void update_left_stack();
        void update_right_stack();
    };

    /* Tensor mean(const std::function<Tensor(const size_t)>& fnc, const std::vector<size_t> batch, const Tensor init = Tensor()); */
    /* double mean(const std::function<double(const size_t)>& fnc, const std::vector<size_t>& batch, const double& init); */
    /* Tensor mean(const std::function<Tensor(const size_t)>& fnc, const std::vector<size_t>& batch, const Tensor& init); */
}
