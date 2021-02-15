#pragma once

#include <armadillo>
#include "autograd.hpp"
#include "common.hpp"
#include "material.hpp"
#include "scene.hpp"

using namespace arma;

namespace drt {

class ScatterResult : public Autograd<rgb> {
public:
    ScatterResult(rgb value, Autograd<rgb> *emission, Autograd<rgb> *brdf_value,
        Autograd<rgb> *radiance, double cos_theta, double pdf)
      : Autograd(value)
      , m_emission(emission)
      , m_brdf_value(brdf_value)
      , m_radiance(radiance)
      , m_cos_theta(cos_theta)
      , m_pdf(pdf)
    { }

    ~ScatterResult()
    {
        // delete m_emission; // HACK: This is usually a variable
        delete m_brdf_value;
        delete m_radiance;
    }

    void backward(rgb weight) override
    {
        // TODO: Double check this
        m_emission->backward(weight);
        m_radiance->backward(m_brdf_value->value() % weight * m_cos_theta / m_pdf);
        m_brdf_value->backward(m_radiance->value() % weight);
    }

private:
    Autograd<rgb> *m_emission;
    Autograd<rgb> *m_brdf_value;
    Autograd<rgb> *m_radiance;
    double m_cos_theta;
    double m_pdf;
};

class Pathtracer {
public:
    Pathtracer(double absorb, int min_bounces)
      : m_absorb(absorb)
      , m_min_bounces(min_bounces)
    { }

    Autograd<rgb> *trace(Scene &scene, vec3 orig, vec3 dir, int depth = 0);

private:
    struct RaycastHit {
        vec3 point;
        vec3 normal;
        Material material;
    };

    bool raycast(Scene& scene, vec3 orig, vec3 dir, RaycastHit& hit)
    {
        double tmin = inf;
        for (auto &surface : scene) {
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

    Autograd<rgb> *scatter(Scene &scene, RaycastHit &hit, vec3 dir_in, int depth)
    {
        double pdf;
        vec3 dir_out = hit.material.brdf->sample(hit.normal, -dir_in, pdf);
        vec3 orig = hit.point + 1e-3*dir_out;
        Autograd<rgb> *emission = hit.material.emission;
        Autograd<rgb> *brdf_value = hit.material.brdf->operator()(hit.normal, -dir_in, dir_out);
        Autograd<rgb> *radiance = trace(scene, orig, dir_out, depth+1);
        double cos_theta = dot(hit.normal, dir_out);
        rgb value = emission->value() + brdf_value->value() % radiance->value() * cos_theta / pdf;
        return new ScatterResult(value, emission, brdf_value, radiance, cos_theta, pdf);
    }

    double m_absorb;
    int m_min_bounces;
};

Autograd<rgb> *Pathtracer::trace(Scene& scene, vec3 orig, vec3 dir, int depth)
{
    if (depth >= m_min_bounces && randu() < m_absorb) {
        return new Constant<rgb>(vec3(fill::zeros));
    }
    RaycastHit hit;
    if (raycast(scene, orig, dir, hit)) {
        return scatter(scene, hit, dir, depth);
    } else {
        return new Constant<rgb>(vec3(fill::zeros));
    }
}

}
