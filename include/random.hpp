#pragma once

#include <cstdlib>

namespace drt { namespace random {

static double uniform()
{
    return double(rand()) / (RAND_MAX - 1);
}

} }
