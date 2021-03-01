#pragma once

#include "bxdf.hpp"
#include "emitter.hpp"
#include "vector.hpp"
#include "shape.hpp"

namespace drt {

using Scene = std::vector<Shape*>;

namespace internal {

Vec3 sample_bxdf(const BxDF *bxdf, Vec3 normal, Vec3 dir_in, double& pdf)
{ return bxdf ? bxdf->sampler(normal, dir_in)->sample(pdf) : Vec3(0); }

Var3 eval_bxdf(const BxDF *bxdf, Vec3 normal, Vec3 dir_in, Vec3 dir_out)
{ return bxdf ? (*bxdf)(normal, dir_in, dir_out) : Vec3(0); }

Var3 emission(const Emitter *emitter)
{ return emitter ? emitter->emission() : Vec3(0); }

} // namespace internal

class Pathtracer {
public:
    Pathtracer(double absorb, int min_bounces)
      : m_absorb(absorb), m_min_bounces(min_bounces) { }

    Var3 trace(const Scene& scene, Vec3 orig, Vec3 dir, int depth = 0);

private:
    struct RaycastHit {
        Vec3 point;
        Vec3 normal;
        BxDF *bxdf;
        Emitter *emitter;
    };

    bool raycast(const Scene& scene, Vec3 orig, Vec3 dir, RaycastHit& hit)
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

    Var3 scatter(const Scene& scene, RaycastHit& hit, Vec3 dir_in, int depth)
    {
        double pdf;
        Var3 diffuse = integrate<double, 3>(
            [=](const Vec3& dir_out)
            {
                Vec3 orig = hit.point + 1e-3*dir_out;
                Var3 brdf_value = internal::eval_bxdf(hit.bxdf, hit.normal, -dir_in, dir_out);
                Var3 radiance = trace(scene, orig, dir_out, depth+1);
                double cos_theta = dot(hit.normal, dir_out);
                return brdf_value * radiance * cos_theta;
            },
            [=](double& pdf)
            {
                return internal::sample_bxdf(hit.bxdf, hit.normal, -dir_in, pdf);
            },
            1,
            false
        );
        Var3 emission = internal::emission(hit.emitter);
        return emission + diffuse;
    }

    double m_absorb;
    int m_min_bounces;
};

Var3 Pathtracer::trace(const Scene& scene, Vec3 orig, Vec3 dir, int depth)
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
