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

    virtual Vector<T, 3, true> operator()(
        const Vector<T, 3>& normal,
        const Vector<T, 3>& dir_in,
        const Vector<T, 3>& dir_out) const = 0;

    virtual Vector<T, 3> sample(const Vector<T, 3>& normal,
                                const Vector<T, 3>& dir_in,
                                double& pdf) const = 0;
};

namespace internal {

template <typename T>
inline void make_frame(const Vector<T, 3>& normal,
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

    Vector<T, 3, true> operator()(const Vector<T, 3>& normal,
                                  const Vector<T, 3>& dir_in,
                                  const Vector<T, 3>& dir_out) const override
    {
        return m_color / pi;
    }

    Vector<T, 3> sample(const Vector<T, 3>& normal,
                        const Vector<T, 3>& dir_in,
                        double& pdf) const override
    {
        double theta = asin(sqrt(random::uniform()));
        double phi = 2 * pi * random::uniform();
        Vector<T, 3> tangent, bitangent;
        internal::make_frame(normal, tangent, bitangent);
        double dir_n = cos(theta);
        double dir_t = cos(phi) * sin(theta);
        double dir_b = sin(phi) * sin(theta);
        pdf = cos(theta) / pi;
        return dir_n*normal + dir_t*tangent + dir_b*bitangent;
    }

private:
    Vector<T, 3, true> m_color;
};

template <typename T>
class SpecularBxDF : public BxDF<T> {
public:
    SpecularBxDF(const Vector<T, 3, true>& color, double exponent)
      : m_color(color), m_exponent(exponent) { }

    Vector<T, 3, true> operator()(const Vector<T, 3>& normal,
                                  const Vector<T, 3>& dir_in,
                                  const Vector<T, 3>& dir_out) const override
    {
        Vector<T, 3> halfway = normalize(dir_in + dir_out);
        double cos_theta = real(dot(normal, halfway));
        double sin_theta = sqrt(1 - cos_theta*cos_theta);
        double factor = (m_exponent + 2) / (2 * pi)
            * pow(cos_theta, m_exponent) * sin_theta;
        return factor * m_color;
    }

    Vector<T, 3> sample(const Vector<T, 3>& normal,
                        const Vector<T, 3>& dir_in,
                        double& pdf) const override
    {
        double theta = acos(sqrt(pow(random::uniform(), 2/(m_exponent+2))));
        double phi = 2 * pi * random::uniform();
        Vector<T, 3> tangent, bitangent;
        internal::make_frame(normal, tangent, bitangent);
        double dir_n = cos(theta);
        double dir_t = cos(phi) * sin(theta);
        double dir_b = sin(phi) * sin(theta);
        Vector<T, 3> halfway = dir_n*normal + dir_t*tangent + dir_b*bitangent;
        if (real(dot(halfway, dir_in)) < 0)
            halfway = -halfway + 2*real(dot(normal, halfway))*normal;
        pdf = (m_exponent + 2) / (2 * pi) *
            pow(cos(theta), m_exponent+1) * sin(theta);
        return -dir_in + 2*real(dot(halfway, dir_in))*halfway;
    }
private:
    Vector<T, 3, true> m_color;
    double m_exponent;
};

template <typename T>
class MirrorBxDF : public BxDF<T> {
public:
    Vector<T, 3, true> operator()(const Vector<T, 3>& normal,
                                  const Vector<T, 3>& dir_in,
                                  const Vector<T, 3>& dir_out) const override
    {
        double cos_theta = real(dot(normal, dir_out));
        return 1 / cos_theta;
    }

    Vector<T, 3> sample(const Vector<T, 3>& normal,
                        const Vector<T, 3>& dir_in,
                        double& pdf) const override
    {
        pdf = 1;
        return -dir_in + 2*real(dot(normal, dir_in))*normal;
    }
};

} // namespace drt
