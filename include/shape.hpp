#pragma once

#include "bxdf.hpp"
#include "constants.hpp"
#include "emitter.hpp"
#include "vector.hpp"

namespace drt {

class Shape {
public:
    Shape(std::shared_ptr<BxDF> bxdf = nullptr,
          std::shared_ptr<Emitter> emitter = nullptr)
      : m_bxdf(bxdf), m_emitter(emitter) { }

    virtual ~Shape() { }

    virtual bool intersect(Vec3 orig, Vec3 dir, double& t) = 0;

    virtual Vec3 normal(Vec3 point) = 0;

    BxDF *bxdf()
    { return m_bxdf.get(); }

    Emitter *emitter()
    { return m_emitter.get(); }

private:
    std::shared_ptr<BxDF> m_bxdf;
    std::shared_ptr<Emitter> m_emitter;
};

class Plane : public Shape {
public:
    Plane(Vec3 normal, double offset,
          std::shared_ptr<BxDF> bxdf = nullptr,
          std::shared_ptr<Emitter> emitter = nullptr)
      : Shape(bxdf, emitter)
      , m_normal(normal)
      , m_offset(offset)
    { }

    bool intersect(Vec3 orig, Vec3 dir, double& t_) override
    {
        double h = dot(orig, m_normal) - m_offset;
        double t = h / dot(dir, -m_normal);
        if (t <= 0.) {
            return false;
        } else {
            t_ = t;
            return true;
        }
    }

    Vec3 normal(Vec3 point) override
    { return m_normal; }

private:
    Vec3 m_normal;
    double m_offset;
};

class Sphere : public Shape {
public:
    Sphere(Vec3 center, double radius,
           std::shared_ptr<BxDF> bxdf = nullptr,
           std::shared_ptr<Emitter> emitter = nullptr)
      : Shape(bxdf, emitter)
      , m_center(center)
      , m_radius(radius)
    { }

    bool intersect(Vec3 orig, Vec3 dir, double& t)
    {
        orig -= m_center;
        double a = 1.;
        double b = 2. * dot(orig, dir);
        double c = dot(orig, orig) - m_radius*m_radius;
        double d = b*b - 4.*a*c;
        if (d < 0.) {
            return false;
        }
        double t1 = (-b - sqrt(d)) / (2. * a);
        double t2 = (-b + sqrt(d)) / (2. * a);
        if (t1 > 0. && t2 > 0.) {
            t = std::min(t1, t2);
            return true;
        } else if (t1 > 0.) {
            t = t1;
            return true;
        } else if (t2 > 0.) {
            t = t2;
            return true;
        } else {
            return false;
        }
    }

    Vec3 normal(Vec3 point)
    { return normalize(point - m_center); }

private:
    Vec3 m_center;
    double m_radius;
};

}
