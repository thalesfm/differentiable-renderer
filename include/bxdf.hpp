#pragma once

#include "constants.hpp"
#include "random.hpp"

namespace drt {

namespace internal {

static void make_frame(Vec3 normal, Vec3& tangent, Vec3& bitangent)
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

class BxDF {
public:
    virtual ~BxDF() { };
    virtual Var3 operator()(Vec3 normal, Vec3 dir_in, Vec3 dir_out) = 0;
    virtual Vec3 sample(Vec3 normal, Vec3 dir_in, double& pdf) = 0;
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

    Vec3 sample(Vec3 normal, Vec3 dir_in, double& pdf) override
    {
        double theta = asin(sqrt(random::uniform()));
        double phi = 2 * pi * random::uniform();
        Vec3 tangent, bitangent;
        internal::make_frame(normal, tangent, bitangent);
        double dir_t = cos(phi) * sin(theta);
        double dir_b = sin(phi) * sin(theta);
        double dir_n = cos(theta);
        pdf = cos(theta) / pi;
        return dir_t*tangent + dir_b*bitangent + dir_n*normal;
    }

private:
    Var3 m_color;
};

}
