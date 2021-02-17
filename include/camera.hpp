#pragma once

#include <armadillo>

using namespace arma;

namespace drt {

class Camera {
public:
    Camera(int width,
           int height,
           double vfov = 1.3963,
           vec3 eye = vec3 {0., 0., 0.},
           vec3 forward = vec3 {0., 0., 1.},
           vec3 right = vec3 {-1., 0., 0.},
           vec3 up = vec3 {0., 1., 0.})
     : m_width(width)
     , m_height(height)
     , m_vfov(vfov)
     , m_eye(eye)
     , m_forward(forward)
     , m_right(right)
     , m_up(up)
    { }

    void look_at(vec3 eye, vec3 at, vec3 up)
    {
        m_eye = eye;
        m_forward = normalise(at - eye);
        m_right = normalise(cross(m_forward, up));
        m_up = cross(m_right, m_forward);
    }

    int width() const
    { return m_width; }

    int height() const
    { return m_height; }

    double aspect() const
    { return double(m_width) / m_height; }

    void pix2ray(double x, double y, vec3 &orig, vec3 &dir)
    {
        orig = m_eye;
        double s = x / m_width;
        double t = y / m_height;
        dir = m_forward;
        dir += (2.*s - 1.) * aspect() * tan(m_vfov / 2.) * m_right;
        dir += (2.*t - 1.) * tan(m_vfov / 2.) * -m_up;
        dir = normalise(dir);
    }
private:
    int m_width;
    int m_height;
    double m_vfov;
    vec3 m_eye;
    vec3 m_forward;
    vec3 m_right;
    vec3 m_up;
};

}
