#pragma once

#include "bxdf.hpp"
#include "emitter.hpp"
#include "vector.hpp"
#include "shape.hpp"

namespace drt {

using Scene = std::vector<Shape*>;

class Pathtracer {
public:
    Pathtracer(double absorb, int min_bounces)
      : m_absorb(absorb), m_min_bounces(min_bounces) { }

    Var3 trace(Scene& scene, Vec3 orig, Vec3 dir, int depth = 0);

private:
    struct RaycastHit {
        Vec3 point;
        Vec3 normal;
        BxDF *bxdf;
        Emitter *emitter;
    };

    bool raycast(Scene& scene, Vec3 orig, Vec3 dir, RaycastHit& hit)
    {
        double tmin = inf;
        for (auto shape : scene) {
            double t;
            if (!shape->intersect(orig, dir, t) || t >= tmin) {
                continue;
            }
            tmin = t;
            hit.point = orig + t*dir;
            hit.normal = shape->normal(hit.point);
            hit.bxdf = shape->bxdf();
            hit.emitter = shape->emitter();
        }
        return !std::isinf(tmin);
    }

    Var3 scatter(Scene& scene, RaycastHit& hit, Vec3 dir_in, int depth)
    {
        Var3 brdf_value;
        Var3 emission;
        Vec3 dir_out;
        Vec3 orig;
        double pdf;
        if (hit.bxdf) {
            auto sampler = hit.bxdf->sampler(hit.normal, -dir_in);
            dir_out = sampler->sample(pdf);
            orig = hit.point + 1e-3*dir_out;
            brdf_value = hit.bxdf->operator()(hit.normal, -dir_in, dir_out);
        } else {
            brdf_value = Vec3(0);
        }
        if (hit.emitter) {
            emission = hit.emitter ? hit.emitter->emission() : Vec3(0);
        } else {
            emission = Vec3(0);
        }
        Var3 radiance = trace(scene, orig, dir_out, depth+1);
        double cos_theta = dot(hit.normal, dir_out);
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
        return scatter(scene, hit, dir, depth);
    } else {
        return Vec3(0);
    }
}

}
