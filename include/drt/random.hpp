#pragma once

#include <cstdlib>

namespace drt { namespace random {

inline double uniform()
{
    return double(rand()) / (RAND_MAX - 1);
}

} }
