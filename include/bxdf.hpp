#pragma once

#include <memory>
#include "complex.hpp"
#include "constants.hpp"
#include "random.hpp"
#include "vector.hpp"

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

namespace internal {

template <typename T>
inline void make_frame(Vector<T, 3> normal,
                       Vector<T, 3>& tangent,
                       Vector<T, 3>& bitangent)
{
    Vector<T, 3> e1 {1., 0., 0.};
    Vector<T, 3> e2 {0., 1., 0.};
    if (std::abs(real(dot(e1, normal))) < std::abs(real(dot(e2, normal))))
        tangent = normalize(e1 - normal*dot(e1, normal));
    else
        tangent = normalize(e2 - normal*dot(e2, normal));
    bitangent = normalize(cross(normal, tangent));
}

} // namespace internal

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
        double theta = asin(sqrt(random::uniform()));
        double phi = 2 * pi * random::uniform();
        Vector<T, 3> tangent, bitangent;
        internal::make_frame(normal, tangent, bitangent);
        double dir_t = cos(phi) * sin(theta);
        double dir_b = sin(phi) * sin(theta);
        double dir_n = cos(theta);
        pdf = cos(theta) / pi;
        return dir_t*tangent + dir_b*bitangent + dir_n*normal;
    }

private:
    Vector<T, 3, true> m_color;
};

template <typename T>
class MirrorBxDF : public BxDF<T> {
public:
    Vector<T, 3, true> operator()(Vector<T, 3> normal,
                                  Vector<T, 3> dir_in,
                                  Vector<T, 3> dir_out) const override
    {
        double cos_theta = real(dot(normal, dir_out));
        return Vector<T, 3, true>(1 / cos_theta,
            [](const Vector<T, 3>& grad) { });
    }

    Vector<T, 3> sample(Vector<T, 3> normal,
                        Vector<T, 3> dir_in,
                        double& pdf) const override
    {
        pdf = 1;
        return -dir_in + 2*real(dot(normal, dir_in))*normal;
    }
};

} // namespace drt
