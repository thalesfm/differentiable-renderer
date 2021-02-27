#pragma once

#include "constants.hpp"
#include "random.hpp"
#include "vector.hpp"

namespace drt {

template <typename T>
class Sampler {
public:
    virtual ~Sampler() { }
    virtual T sample(double& pdf) = 0;
};

namespace internal {

inline void make_frame(Vec3 normal, Vec3& tangent, Vec3& bitangent)
{
    Vec3 e1 {1., 0., 0.};
    Vec3 e2 {0., 1., 0.};
    if (abs(dot(e1, normal)) < abs(dot(e2, normal))) {
        tangent = normalize(e1 - normal*dot(e1, normal));
    } else {
        tangent = normalize(e2 - normal*dot(e2, normal));
    }
    bitangent = normalize(cross(normal, tangent));
}

} // namespace internal

class UniformHemisphereSampler : public Sampler<Vec3> {
public:
    UniformHemisphereSampler(Vec3 normal) : m_normal(normal) { }

    Vec3 sample(double& pdf) override
    {
        double theta = acos(random::uniform());
        double phi = 2 * pi * random::uniform();
        Vec3 tangent, bitangent;
        internal::make_frame(m_normal, tangent, bitangent);
        double dir_t = cos(phi) * sin(theta);
        double dir_b = sin(phi) * sin(theta);
        double dir_n = cos(theta);
        pdf = 1 / (2 * pi);
        return dir_t*tangent + dir_b*bitangent + dir_n*m_normal;
    }

private:
    Vec3 m_normal;
};

class CosineWeightedHemisphereSampler : public Sampler<Vec3> {
public:
    CosineWeightedHemisphereSampler(Vec3 normal) : m_normal(normal) { }

    Vec3 sample(double& pdf) override
    {
        double theta = asin(sqrt(random::uniform()));
        double phi = 2 * pi * random::uniform();
        Vec3 tangent, bitangent;
        internal::make_frame(m_normal, tangent, bitangent);
        double dir_t = cos(phi) * sin(theta);
        double dir_b = sin(phi) * sin(theta);
        double dir_n = cos(theta);
        pdf = cos(theta) / pi;
        return dir_t*tangent + dir_b*bitangent + dir_n*m_normal;
    }

private:
    Vec3 m_normal;
};

}
