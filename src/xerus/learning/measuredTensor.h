#pragma once

#include <xerus.h>


namespace xerus {
    template <class InternalTensor>
    class MeasuredTensor {
    public:
        virtual size_t get_numComponents() const = 0;
        virtual size_t get_numMeasurements() const = 0;
        virtual Tensor value(const size_t measIdx) const = 0;
        virtual TensorNetwork gradient(const size_t measIdx) const = 0;
        virtual const InternalTensor& get_tensor() const = 0;
        virtual const Tensor& get_core() const = 0;
        virtual void set_core(const Tensor& val) = 0;
        virtual size_t get_corePosition() const = 0;
        virtual void set_corePosition(const size_t pos) = 0;
    };
}
