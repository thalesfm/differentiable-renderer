#pragma once

#include "complex.hpp"
#include "constants.hpp"
#include "random.hpp"
#include "vector.hpp"

namespace drt {

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
inline Vector<T, 3> uniform_hemisphere(Vector<T, 3> normal, double& pdf)
{
    double theta = acos(random::uniform());
    double phi = 2 * pi * random::uniform();
    Vector<T, 3> tangent, bitangent;
    internal::make_frame(normal, tangent, bitangent);
    double dir_t = cos(phi) * sin(theta);
    double dir_b = sin(phi) * sin(theta);
    double dir_n = cos(theta);
    pdf = 1 / (2 * pi);
    return dir_t*tangent + dir_b*bitangent + dir_n*normal;
}

template <typename T>
inline Vector<T, 3> cosine_weighted_hemisphere(Vector<T, 3> normal, double& pdf)
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

} // namespace drt
