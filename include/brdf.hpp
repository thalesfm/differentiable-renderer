#pragma once

#include <tuple>
#include <armadillo>
#include "autograd.hpp"
#include "common.hpp"

using namespace arma;

namespace drt {

class BRDF {
public:
    virtual ~BRDF() { };
    virtual Autograd<rgb> *operator()(vec3 normal, vec3 dir_in, vec3 dir_out) = 0;
    virtual vec3 sample(vec3 normal, vec3 dir_in, double &pdf) = 0;
};

class BlackBRDFResult : Autograd<rgb> {
public:
    friend class BlackBRDF;

    void backward(rgb weight) override
    { }

private:
    BlackBRDFResult(rgb value)
      : Autograd(value)
    { }
};

class BlackBRDF : public BRDF {
public:
    Autograd<vec3> *operator()(vec3 normal, vec3 dir_in, vec3 dir_out) override
    {
        rgb value =  vec3(fill::zeros);
        return new BlackBRDFResult(value);
    }

    vec3 sample(vec3 normal, vec3 dir_in, double &pdf) override
    {
        pdf = 1.;
        return vec3(fill::zeros);
    }
};

class DiffuseBRDFResult : Autograd<rgb> {
public:
    friend class DiffuseBRDF;

    void backward(rgb weight) override
    { m_color.backward(weight / pi); }

private:
    DiffuseBRDFResult(rgb value, Variable<rgb>& color)
      : Autograd(value)
      , m_color(color)
    { }

    Variable<rgb>& m_color;
};

class DiffuseBRDF : public BRDF {
public:
    DiffuseBRDF(Variable<rgb>& color)
      : m_color(color)
    { }

    vec3 sample(vec3 normal, vec3 dir_in, double &pdf) override
    {
        double theta = acos(sqrt(randu()));
        double phi = 2*pi*randu();
        vec3 tangent, bitangent;
        std::tie(tangent, bitangent) = make_frame(normal);
        double dir_t = cos(phi) * cos(theta);
        double dir_b = sin(phi) * cos(theta);
        double dir_n = sin(theta);
        pdf = cos(pi/2 - theta);
        return dir_t*tangent + dir_b*bitangent + dir_n*normal;
    }

    Autograd<rgb> *operator()(vec3 normal, vec3 dir_in, vec3 dir_out) override
    {
        rgb value = m_color.value() / pi;
        return new DiffuseBRDFResult(value, m_color);
    }

private:
    Variable<rgb>& m_color;
};

}
