#pragma once

#include <tuple>
#include <armadillo>

using namespace arma;

namespace drt {

using rgb = arma::vec3;

const double pi = arma::datum::pi;
const double inf = arma::datum::inf;

const rgb white {1., 1., 1.};
const rgb black {0., 0., 0.};
const rgb red {1., 0., 0.};
const rgb green {0., 1., 0.};
const rgb blue {0., 0., 1.};

static std::tuple<vec3, vec3> make_frame(vec3 normal)
{
    vec3 e1 {1., 0., 0.};
    vec3 e2 {0., 1., 0.};
    vec3 tangent;
    if (abs(dot(e1, normal)) < abs(dot(e2, normal))) {
        tangent = normalise(e1 - normal*dot(e1, normal));
    } else {
        tangent = normalise(e2 - normal*dot(e2, normal));
    }
    vec3 bitangent = normalise(cross(normal, tangent));
    return std::make_tuple(tangent, bitangent);
}

}
