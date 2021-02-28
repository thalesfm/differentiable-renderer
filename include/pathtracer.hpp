#pragma once

#include "bxdf.hpp"
#include "emitter.hpp"
#include "vector.hpp"
#include "shape.hpp"

namespace drt {

using Scene = std::vector<Shape*>;

namespace internal {

Vec3 sample_bxdf(BxDF *bxdf, Vec3 normal, Vec3 dir_in, double& pdf)
{ return bxdf ? bxdf->sampler(normal, dir_in)->sample(pdf) : Vec3(0); }

Var3 eval_bxdf(BxDF *bxdf, Vec3 normal, Vec3 dir_in, Vec3 dir_out)
{ return bxdf ? (*bxdf)(normal, dir_in, dir_out) : Vec3(0); }

Var3 emission(Emitter *emitter)
{ return emitter ? emitter->emission() : Vec3(0); }

} // namespace internal

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
        double pdf;
        Vec3 dir_out = internal::sample_bxdf(hit.bxdf, hit.normal, -dir_in, pdf);
        Vec3 orig = hit.point + 1e-3*dir_out;
        Var3 brdf_value = internal::eval_bxdf(hit.bxdf, hit.normal, -dir_in, dir_out);
        Var3 emission = internal::emission(hit.emitter);
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
    double p = depth >= m_min_bounces ? (1 - m_absorb) : 1;
    RaycastHit hit;
    if (raycast(scene, orig, dir, hit)) {
        return scatter(scene, hit, dir, depth) / p;
    } else {
        return Vec3(0);
    }
}

}
