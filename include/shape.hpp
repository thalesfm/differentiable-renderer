#pragma once

#include <cmath>
#include "bxdf.hpp"
#include "complex.hpp"
#include "constants.hpp"
#include "emitter.hpp"
#include "vector.hpp"

namespace drt {

template <typename T>
class Shape {
public:
    Shape(std::shared_ptr<BxDF<T>> bxdf = nullptr,
          std::shared_ptr<Emitter<T>> emitter = nullptr)
      : m_bxdf(bxdf), m_emitter(emitter) { }

    virtual ~Shape() { }

    virtual bool intersect(Vector<T, 3> orig,
                           Vector<T, 3> dir,
                           double& t) const = 0;

    virtual Vector<T, 3> normal(Vector<T, 3> point) const = 0;

    BxDF<T> *bxdf()
    { return m_bxdf.get(); }

    Emitter<T> *emitter()
    { return m_emitter.get(); }

private:
    std::shared_ptr<BxDF<T>> m_bxdf;
    std::shared_ptr<Emitter<T>> m_emitter;
};

template <typename T>
class Plane : public Shape<T> {
public:
    Plane(Vector<T, 3> normal,
          double offset,
          std::shared_ptr<BxDF<T>> bxdf = nullptr,
          std::shared_ptr<Emitter<T>> emitter = nullptr)
      : Shape<T>(bxdf, emitter)
      , m_normal(normal)
      , m_offset(offset)
    { }

    bool intersect(Vector<T, 3> orig,
                   Vector<T, 3> dir,
                   double& t) const override
    {
        double h = real(dot(orig, m_normal)) - m_offset;
        t = h / real(dot(dir, -m_normal));
        return t > 0;
    }

    Vector<T, 3> normal(Vector<T, 3> point) const override
    { return m_normal; }

private:
    Vector<T, 3> m_normal;
    double m_offset;
};

template <typename T>
class Sphere : public Shape<T> {
public:
    Sphere(Vector<T, 3> center,
           double radius,
           std::shared_ptr<BxDF<T>> bxdf = nullptr,
           std::shared_ptr<Emitter<T>> emitter = nullptr)
      : Shape<T>(bxdf, emitter)
      , m_center(center)
      , m_radius(radius)
    { }

    bool intersect(Vector<T, 3> orig,
                   Vector<T, 3> dir,
                   double& t) const override
    {
        orig -= m_center;
        double a = 1;
        double b = 2 * real(dot(orig, dir));
        double c = real(dot(orig, orig)) - m_radius*m_radius;
        double d = b*b - 4*a*c;
        if (d < 0)
            return false;
        double t1 = (-b - std::sqrt(d)) / (2 * a);
        double t2 = (-b + std::sqrt(d)) / (2 * a);
        if (t1 > 0 && t2 > 0) {
            t = std::min(t1, t2);
            return true;
        } else if (t1 > 0) {
            t = t1;
            return true;
        } else if (t2 > 0) {
            t = t2;
            return true;
        } else {
            return false;
        }
    }

    Vector<T, 3> normal(Vector<T, 3> point) const override
    { return normalize(point - m_center); }

private:
    Vector<T, 3> m_center;
    double m_radius;
};

}
