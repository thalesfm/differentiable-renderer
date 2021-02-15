#pragma once

#include <armadillo>
#include "common.hpp"

using namespace arma;

namespace drt {

class Shape {
public:
    virtual ~Shape() { }
    virtual bool intersect(vec3 orig, vec3 dir, double& t) = 0;
    virtual vec3 normal(vec3 point) = 0;
};

class Plane : public Shape {
public:
    Plane(vec3 normal, double offset)
      : m_normal(normal)
      , m_offset(offset)
    { }

    bool intersect(vec3 orig, vec3 dir, double& t_)
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

    vec3 normal(vec3 point)
    {
        return m_normal;
    }
private:
    vec3 m_normal;
    double m_offset;
};

class Sphere : public Shape {
public:
    Sphere(vec3 center, double radius)
      : m_center(center)
      , m_radius(radius)
    { }

    bool intersect(vec3 orig, vec3 dir, double& t)
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

    vec3 normal(vec3 point)
    {
        return normalise(point - m_center);
    }
private:
    vec3 m_center;
    double m_radius;
};

}
