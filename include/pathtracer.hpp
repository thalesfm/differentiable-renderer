#pragma once

#include "material.hpp"
#include "vector.hpp"
#include "scene.hpp"

namespace drt {

class Pathtracer {
public:
    Pathtracer(double absorb, int min_bounces)
      : m_absorb(absorb), m_min_bounces(min_bounces) { }

    Var3 trace(Scene& scene, Vec3 orig, Vec3 dir, int depth = 0);

private:
    struct RaycastHit {
        Vec3 point;
        Vec3 normal;
        Material material;
    };

    bool raycast(Scene& scene, Vec3 orig, Vec3 dir, RaycastHit& hit)
    {
        double tmin = inf;
        for (auto& surface : scene) {
            Shape *shape;
            Material material;
            std::tie(shape, material) = surface;
            double t;
            if (!shape->intersect(orig, dir, t) || t >= tmin) {
                continue;
            }
            tmin = t;
            hit.point = orig + t*dir;
            hit.normal = shape->normal(hit.point);
            hit.material = material;
        }
        return !std::isinf(tmin);
    }

    Var3 scatter(Scene& scene, RaycastHit& hit, Vec3 dir_in, int depth)
    {
        double pdf;
        Vec3 dir_out = hit.material.bxdf->sample(hit.normal, -dir_in, pdf);
        Vec3 orig = hit.point + 1e-3*dir_out;
        Var3 emission = hit.material.emission;
        Var3 brdf_value = hit.material.bxdf->operator()(hit.normal, -dir_in, dir_out);
        Var3 radiance = trace(scene, orig, dir_out, depth+1);
        double cos_theta = dot(hit.normal, dir_out);
        // std::cout << "\temission = " << emission << std::endl;
        // std::cout << "\tbrdf_value = " << brdf_value << std::endl;
        // std::cout << "\tradiance = " << radiance << std::endl;
        return emission + brdf_value * radiance * cos_theta / pdf;
    }

    double m_absorb;
    int m_min_bounces;
};

Var3 Pathtracer::trace(Scene& scene, Vec3 orig, Vec3 dir, int depth)
{
    if (depth >= m_min_bounces && random::uniform() < m_absorb) {
        return Vec3(0);
    }
    RaycastHit hit;
    if (raycast(scene, orig, dir, hit)) {
        // std::cout << "Scatter! (depth =" << depth << ")" << std::endl;
        return scatter(scene, hit, dir, depth);
    } else {
        return Vec3(0);
    }
}

}
