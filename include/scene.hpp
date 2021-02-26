#pragma once

#include "shape.hpp"
#include "material.hpp"

namespace drt {

class Scene {
private:
    std::vector<std::tuple<Shape*, Material>> m_surfaces;

public:
    decltype(m_surfaces.begin()) begin()
    { return m_surfaces.begin(); }

    decltype(m_surfaces.end()) end()
    { return m_surfaces.end(); }

    decltype(m_surfaces.rbegin()) rbegin()
    { return m_surfaces.rbegin(); }

    decltype(m_surfaces.rend()) rend()
    { return m_surfaces.rend(); }

    void add(Shape &shape, Material material)
    { m_surfaces.emplace_back(&shape, material); }
};

}
