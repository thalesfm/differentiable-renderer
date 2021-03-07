#pragma once

#include <cstddef>
#include <type_traits>
#include "vector.hpp"

namespace drt {

namespace internal {

template <typename T, std::size_t N, typename Forward, typename Sampler>
struct IntegrateBackward {
    void operator()(const Vector<T, N>& grad) const
    {
        for (std::size_t i = 0; i < n_samples; ++i) {
            auto [sample, pdf] = sampler();
            forward(sample).backward(grad / pdf);
        }
    }

    typename std::decay<Forward>::type forward;
    typename std::decay<Sampler>::type sampler;
    std::size_t n_samples;
};

template <typename T, std::size_t N, typename Forward, typename Sampler>
inline Vector<T, N, true> integrate_biased(const Forward& forward,
                                           const Sampler& sampler,
                                           std::size_t n_samples)
{
    Vector<T, N, true> r(0);
    for (std::size_t i = 0; i < n_samples; ++i) {
        auto [sample, pdf] = sampler();
        r += forward(sample) / pdf;
    }
    return r;
}

template <typename T, std::size_t N, typename Forward, typename Sampler>
inline Vector<T, N, true> integrate_unbiased(const Forward& forward,
                                             const Sampler& sampler,
                                             std::size_t n_samples)
{
    Vector<T, N> r(0);
    for (std::size_t i = 0; i < n_samples; ++i) {
        auto [sample, pdf] = sampler();
        r += forward(sample).detach() / pdf;
    }
    return Vector<T, N, true>(r,
        IntegrateBackward<T, N, Forward, Sampler>
            {forward, sampler, n_samples});
}

} // namespace internal

template <typename T, std::size_t N, typename Forward, typename Sampler>
inline Vector<T, N, true> integrate(const Forward& forward,
                                    const Sampler& sampler,
                                    std::size_t n_samples,
                                    bool unbiased = false)
{
    if (unbiased)
        return internal::integrate_unbiased<T, N>(forward, sampler, n_samples);
    else
        return internal::integrate_biased<T, N>(forward, sampler, n_samples);
}

} // namespace drt
