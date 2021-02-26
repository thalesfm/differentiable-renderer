#pragma once

#include "vector.hpp"

namespace drt {

class Camera {
public:
    Camera(int width,
           int height,
           double vfov = 1.3963,
           Vec3 eye = Vec3(0),
           Vec3 forward = Vec3 {0., 0., 1.},
           Vec3 right = Vec3 {-1., 0., 0.},
           Vec3 up = Vec3 {0., 1., 0.})
     : m_width(width)
     , m_height(height)
     , m_vfov(vfov)
     , m_eye(eye)
     , m_forward(forward)
     , m_right(right)
     , m_up(up)
    { }

    void look_at(Vec3 eye, Vec3 at, Vec3 up)
    {
        m_eye = eye;
        m_forward = normalize(at - eye);
        m_right = normalize(cross(m_forward, up));
        m_up = cross(m_right, m_forward);
    }

    int width() const
    { return m_width; }

    int height() const
    { return m_height; }

    double aspect() const
    { return double(m_width) / m_height; }

    void pix2ray(double x, double y, Vec3& orig, Vec3& dir)
    {
        orig = m_eye;
        double s = x / m_width;
        double t = y / m_height;
        dir = m_forward;
        dir += (2.*s - 1.) * aspect() * tan(m_vfov / 2.) * m_right;
        dir += (2.*t - 1.) * tan(m_vfov / 2.) * -m_up;
        dir = normalize(dir);
    }
private:
    int m_width;
    int m_height;
    double m_vfov;
    Vec3 m_eye;
    Vec3 m_forward;
    Vec3 m_right;
    Vec3 m_up;
};

}
