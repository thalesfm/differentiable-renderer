#pragma once

#include "bxdf.hpp"
#include "vector.hpp"

namespace drt {

struct Material {
    BxDF *bxdf;
    Var3 emission;
};

}
