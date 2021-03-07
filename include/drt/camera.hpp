#pragma once

#include <cstddef>
#include <tuple>
#include "random.hpp"
#include "vector.hpp"

namespace drt {

template <typename T>
class Camera {
public:
    Camera(std::size_t width,
           std::size_t height,
           double vfov = 1.3963,
           Vector<T, 3> eye = Vector<T, 3>(0),
           Vector<T, 3> forward = Vector<T, 3>{0, 0, -1},
           Vector<T, 3> right = Vector<T, 3>{1, 0, 0},
           Vector<T, 3> up = Vector<T, 3>{0, 1, 0})
     : m_width(width)
     , m_height(height)
     , m_vfov(vfov)
     , m_eye(eye)
     , m_forward(forward)
     , m_right(right)
     , m_up(up)
    { }

    void look_at(Vector<T, 3> eye,
                 Vector<T, 3> at,
                 Vector<T, 3> up = Vector<T, 3>{0, 1, 0})
    {
        m_eye = eye;
        m_forward = normalize(at - eye);
        m_right = normalize(cross(m_forward, up));
        m_up = cross(m_right, m_forward);
    }

    std::size_t width() const
    { return m_width; }

    std::size_t height() const
    { return m_height; }

    Vector<T, 3> eye() const
    { return m_eye; }

    double aspect() const
    { return double(m_width) / m_height; }

    std::tuple<Vector<T, 3>, double> sample(std::size_t x, std::size_t y) const
    {
        double s = (x + random::uniform()) / m_width;
        double t = (y + random::uniform()) / m_height;
        Vector<T, 3> dir = m_forward;
        dir += (2.*s - 1.) * aspect() * tan(m_vfov / 2.) * m_right;
        dir += (2.*t - 1.) * tan(m_vfov / 2.) * -m_up;
        dir = normalize(dir);
        return std::make_tuple(dir, 1);
    }

private:
    std::size_t m_width;
    std::size_t m_height;
    double m_vfov;
    Vector<T, 3> m_eye;
    Vector<T, 3> m_forward;
    Vector<T, 3> m_right;
    Vector<T, 3> m_up;
};

} // namespace drt
