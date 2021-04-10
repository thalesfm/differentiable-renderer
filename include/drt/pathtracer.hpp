#pragma once

#include <tuple>
#include "bxdf.hpp"
#include "emitter.hpp"
#include "vector.hpp"
#include "shape.hpp"
#include <vector>

namespace drt {

template <typename T>
using Scene = std::vector<Shape<T>*>;

namespace internal {

template <typename T>
std::tuple<Vector<T, 3>, double> sample_bxdf(
    const BxDF<T> *bxdf,
    Vector<T, 3> normal,
    Vector<T, 3> dir_in)
{
    if (bxdf)
        return bxdf->sample(normal, dir_in);
    else
        return std::make_tuple(Vector<T, 3>(0), 1);
}

template <typename T>
Vector<T, 3, true> eval_bxdf(
    const BxDF<T> *bxdf,
    Vector<T, 3> normal,
    Vector<T, 3> dir_in,
    Vector<T, 3> dir_out)
{
    if (bxdf)
        return (*bxdf)(normal, dir_in, dir_out);
    else
        return Vector<T, 3>(0);
}

template <typename T>
Vector<T, 3, true> emission(const Emitter<T> *emitter)
{
    if (emitter)
        return emitter->emission();
    else
        return Vector<T, 3>(0);
}

} // namespace internal

template <typename T>
class Pathtracer {
public:
    Pathtracer(double absorb, std::size_t min_bounces)
      : m_absorb(absorb), m_min_bounces(min_bounces) { }

    Vector<T, 3, true> trace(const Scene<T>& scene,
                             Vector<T, 3> orig,
                             Vector<T, 3> dir,
                             std::size_t depth = 0) const;

private:
    struct RaycastHit {
        Vector<T, 3> point;
        Vector<T, 3> normal;
        BxDF<T> *bxdf;
        Emitter<T> *emitter;
    };

    bool raycast(const Scene<T>& scene,
                 Vector<T, 3> orig,
                 Vector<T, 3> dir,
                 RaycastHit& hit) const
    {
        double tmin = inf;
        for (auto shape : scene) {
            double t;
            if (!shape->intersect(orig, dir, t) || t >= tmin)
                continue;
            tmin = t;
            hit.point = orig + t*dir;
            hit.normal = shape->normal(hit.point);
            hit.bxdf = shape->bxdf();
            hit.emitter = shape->emitter();
        }
        return !std::isinf(tmin);
    }

    Vector<T, 3, true> scatter(const Scene<T>& scene,
                               RaycastHit& hit,
                               Vector<T, 3> dir_in,
                               std::size_t depth) const
    {
        Vector<T, 3, true> diffuse = integrate<T, 3>(
            [=](const Vector<T, 3>& dir_out)
            {
                Vector<T, 3> orig = hit.point + 1e-3*dir_out;
                Vector<T, 3, true> brdf_value = internal::eval_bxdf(
                    hit.bxdf, hit.normal, -dir_in, dir_out);
                Vector<T, 3, true> radiance = trace(scene, orig, dir_out, depth+1);
                double cos_theta = dot(hit.normal, dir_out);
                return brdf_value * radiance * cos_theta;
            },
            [=]()
            {
                return internal::sample_bxdf(hit.bxdf, hit.normal, -dir_in);
            },
            1,
            false
        );
        Vector<T, 3, true> emission = internal::emission(hit.emitter);
        return emission + diffuse;
    }

    double m_absorb;
    std::size_t m_min_bounces;
};

template <typename T>
Vector<T, 3, true> Pathtracer<T>::trace(const Scene<T>& scene,
                                        Vector<T, 3> orig,
                                        Vector<T, 3> dir,
                                        std::size_t depth) const

{
    if (depth >= m_min_bounces && random::uniform() < m_absorb)
        return Vector<T, 3>(0);
    double p = depth >= m_min_bounces ? (1 - m_absorb) : 1;
    RaycastHit hit;
    if (raycast(scene, orig, dir, hit))
        return scatter(scene, hit, dir, depth) / p;
    else
        return Vector<T, 3>(0);
}

}
