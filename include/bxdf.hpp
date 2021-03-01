#pragma once

#include <memory>
#include "sampler.hpp"

namespace drt {

template <typename T>
class BxDF {
public:
    virtual ~BxDF() { };

    virtual Vector<T, 3, true> operator()(Vector<T, 3> normal,
                                          Vector<T, 3> dir_in,
                                          Vector<T, 3> dir_out) const = 0;

    virtual Vector<T, 3> sample(Vector<T, 3> normal,
                                Vector<T, 3> dir_in,
                                double& pdf) const = 0;
};

template <typename T>
class DiffuseBxDF : public BxDF<T> {
public:
    DiffuseBxDF(const Vector<T, 3, true>& color) : m_color(color) { }

    Vector<T, 3, true> operator()(Vector<T, 3> normal,
                                  Vector<T, 3> dir_in,
                                  Vector<T, 3> dir_out) const override
    {
        return Vector<T, 3, true>(m_color.detach() / pi,
            [=](const Vector<T, 3>& grad) { m_color.backward(grad / pi); });
    }

    Vector<T, 3> sample(Vector<T, 3> normal,
                        Vector<T, 3> dir_in,
                        double& pdf) const override
    {
        return cosine_weighted_hemisphere(normal, pdf);
    }

private:
    Vector<T, 3, true> m_color;
};

}
