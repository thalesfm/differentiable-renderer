#pragma once

#include <armadillo>
#include "autograd.hpp"
#include "brdf.hpp"
#include "common.hpp"

using namespace arma;

namespace drt {

struct Material {
    BRDF *brdf;
    Autograd<rgb> *emission;
};

}
