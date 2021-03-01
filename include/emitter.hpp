#pragma once

#include "vector.hpp"

namespace drt {

class Emitter {
public:
    virtual ~Emitter() { }
    virtual Var3 emission() const = 0;
};

class AreaEmitter : public Emitter {
public:
    AreaEmitter(Var3 emission) : m_emission(emission) { }

    Var3 emission() const override
    { return m_emission; }

private:
    Var3 m_emission;
};

}
