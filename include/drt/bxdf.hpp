#pragma once

#include <array>
#include <memory>
#include <tuple>
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

    virtual std::tuple<Vector<T, 3>, double> sample(
        const Vector<T, 3>& normal,
        const Vector<T, 3>& dir_in) const = 0;
};

namespace internal {

template <typename T>
inline std::array<Vector<T, 3>, 3> make_frame(const Vector<T, 3>& normal)
{
    Vector<T, 3> e1 {1., 0., 0.};
    Vector<T, 3> e2 {0., 1., 0.};
    Vector<T, 3> tangent;
    if (std::abs(real(dot(e1, normal))) < std::abs(real(dot(e2, normal))))
        tangent = normalize(e1 - normal*dot(e1, normal));
    else
        tangent = normalize(e2 - normal*dot(e2, normal));
    auto bitangent = normalize(cross(normal, tangent));
    return {tangent, bitangent, normal};
}

template <typename T>
inline Vector<T, 3> angle_to_dir(
    double theta, double phi,
    const std::array<Vector<T, 3>, 3>& frame)
{
    double x = cos(phi) * sin(theta);
    double y = sin(phi) * sin(theta);
    double z = cos(theta);
    return x*frame[0] + y*frame[1] + z*frame[2];
}

} // namespace internal

template <typename T>
class DiffuseBxDF : public BxDF<T> {
public:
    DiffuseBxDF(const Vector<T, 3, true>& color)
      : m_color(color)
    { }

    Vector<T, 3, true> operator()(
        const Vector<T, 3>& normal,
        const Vector<T, 3>& dir_in,
        const Vector<T, 3>& dir_out) const override
    { return m_color / pi; }

    std::tuple<Vector<T, 3>, double> sample(
        const Vector<T, 3>& normal,
        const Vector<T, 3>& dir_in) const override
    {
        double theta = asin(sqrt(random::uniform()));
        double phi = 2 * pi * random::uniform();
        auto frame = internal::make_frame(normal);
        auto dir = internal::angle_to_dir(theta, phi, frame);
        double pdf = cos(theta) / pi;
        return std::make_tuple(dir, pdf);
    }

private:
    Vector<T, 3, true> m_color;
};

template <typename T>
class SpecularBxDF : public BxDF<T> {
public:
    SpecularBxDF(const Vector<T, 3, true>& color, double exponent)
      : m_color(color)
      , m_exponent(exponent)
    { }

    Vector<T, 3, true> operator()(
        const Vector<T, 3>& normal,
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

    std::tuple<Vector<T, 3>, double> sample(
        const Vector<T, 3>& normal,
        const Vector<T, 3>& dir_in) const override
    {
        double theta = acos(sqrt(pow(random::uniform(), 2/(m_exponent+2))));
        double phi = 2 * pi * random::uniform();
        auto frame = internal::make_frame(normal);
        auto halfway = internal::angle_to_dir(theta, phi, frame);
        if (real(dot(halfway, dir_in)) < 0)
            halfway = reflect(halfway, normal);
        auto dir = reflect(dir_in, halfway);
        double pdf = (m_exponent + 2) / (2 * pi) *
            pow(cos(theta), m_exponent+1) * sin(theta);
        return std::make_tuple(dir, pdf);
    }
private:
    Vector<T, 3, true> m_color;
    double m_exponent;
};

template <typename T>
class MirrorBxDF : public BxDF<T> {
public:
    Vector<T, 3, true> operator()(
        const Vector<T, 3>& normal,
        const Vector<T, 3>& dir_in,
        const Vector<T, 3>& dir_out) const override
    {
        double cos_theta = real(dot(normal, dir_out));
        return 1 / cos_theta;
    }

    std::tuple<Vector<T, 3>, double> sample(
        const Vector<T, 3>& normal,
        const Vector<T, 3>& dir_in) const override
    {
        return std::make_tuple(reflect(dir_in, normal), 1);
    }
};

} // namespace drt
