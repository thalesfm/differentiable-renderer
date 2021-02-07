#include <iostream>
#include <armadillo>

using namespace arma;

struct Ray {
    vec3 orig;
    vec3 dir;
};

struct RayHit {
    vec3 point;
    vec3 normal;
};

double closesthit(arma::vec ts)
{
    // TODO
}

struct Sphere {
    vec3 center;
    double radius;

    bool intersect(Ray ray, RayHit *hit)
    {
        // Find quadratic roots
        arma::vec3 orig_(ray.orig - center);
        arma::vec3 poly;
        poly(0) = 1.;
        poly(1) = 2. * arma::dot(orig_, ray.dir);
        poly(2) = arma::dot(orig_, orig_) - radius*radius;
        arma::cx_vec cx_roots = arma::roots(poly);

        if (!imag(cx_roots).is_zero()) {
            return false;
        }

        // Determine closest hit
        arma::vec roots = arma::real(cx_roots);
        if (!arma::any(roots) > 0.) {
            return false;
        }

        double t;
        if (roots(0) > 0. && roots(1) > 0.) {
            t = arma::min(roots);
        } else if (roots(0) > 0.) {
            t = roots(0);
        } else {
            t = roots(1);
        }

        if (hit == nullptr) {
            return true;
        }

        // Fill in hit info
        hit->point = ray.orig + t*ray.dir;
        hit->normal = normal(hit->point);
        return true;
    }

    arma::vec3 normal(arma::vec3 point)
    {
        return arma::normalise(point - center);
    }
};

int main(int argc, const char *argv[])
{
    RayHit hit;
    Ray ray1 {vec3{0., 0., 0.}, normalise(vec3{0.1, 0., 1.})};
    Ray ray2 {vec3{0., 0., 0.}, vec3{1., 0., 0.}};
    Sphere sphere {vec3{0., 0., 3.}, 1.};

    std::cout << sphere.intersect(ray1, &hit) << std::endl;
    std::cout << "point:\n" << hit.point << std::endl;
    std::cout << "normal:\n" << hit.normal << std::endl;
    std::cout << sphere.intersect(ray2, &hit) << std::endl;
    return 0;
}
