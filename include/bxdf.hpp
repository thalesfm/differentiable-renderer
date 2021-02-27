#pragma once

#include <memory>
#include "sampler.hpp"

namespace drt {

class BxDF {
public:
    virtual ~BxDF() { };
    virtual Var3 operator()(Vec3 normal, Vec3 dir_in, Vec3 dir_out) = 0;
    virtual std::unique_ptr<Sampler<Vec3>> sampler(Vec3 normal, Vec3 dir_in) = 0;
};

class DiffuseBxDF : public BxDF {
public:
    DiffuseBxDF(const Var3& color) : m_color(color) { }

    Var3 operator()(Vec3 normal, Vec3 dir_in, Vec3 dir_out) override
    {
        return Var3(m_color.detach() / pi, [=](const Vec3& grad) {
            m_color.backward(grad / pi);
        });
    }

    std::unique_ptr<Sampler<Vec3>> sampler(Vec3 normal, Vec3 dir_in)
    {
        return std::unique_ptr<Sampler<Vec3>>(
            new CosineWeightedHemisphereSampler(normal));
    }

private:
    Var3 m_color;
};

}
