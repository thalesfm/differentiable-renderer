#pragma once

#include "vector.hpp"

namespace drt {

template <typename T>
class Emitter {
public:
    virtual ~Emitter() { }

    virtual Vector<T, 3, true> emission() const = 0;
};

template <typename T>
class AreaEmitter : public Emitter<T> {
public:
    AreaEmitter(Vector<T, 3, true> emission) : m_emission(emission) { }

    Vector<T, 3, true> emission() const override
    { return m_emission; }

private:
    Vector<T, 3, true> m_emission;
};

}
