#pragma once

#include <cstddef>
#include <type_traits>
#include "vector.hpp"

namespace drt {

namespace internal {

template <typename T, std::size_t N, typename Forward, typename Sampler>
struct IntegrateBackward {
    void operator()(const Vector<T, N>& grad)
    {
        for (std::size_t i = 0; i < n_samples; ++i)
            forward(sampler()).backward(grad / n_samples);
    }

    typename std::decay<Forward>::type forward;
    typename std::decay<Sampler>::type sampler;
    std::size_t n_samples;
};

} // namespace internal

template <typename T, std::size_t N, typename Forward, typename Sampler>
inline Vector<T, N, true> integrate(const Forward& forward,
                                    const Sampler& sampler,
                                    std::size_t n_samples)
{
    Vector<T, N> r(0);
    for (std::size_t i = 0; i < n_samples; ++i)
        r += forward(sampler()).detach() / n_samples;
    return Vector<T, N, true>(r,
        internal::IntegrateBackward<T, N, Forward, Sampler>
            {forward, sampler, n_samples});
}

} // namespace drt
