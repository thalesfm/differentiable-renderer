#include <cmath>
#include <cstdio>
#include <armadillo>

const double pi = arma::datum::pi;
const double inf = arma::datum::inf;

struct RayHit {
    arma::vec3 point;
    arma::vec3 normal;
};

class Shape {
public:
    virtual double raycast(arma::vec3 orig, arma::vec3 dir) = 0;
    virtual arma::vec3 normal(arma::vec3 point) = 0;
};

class Sphere : public Shape {
public:
    Sphere(arma::vec3 center, double radius)
      : m_center(center), m_radius(radius)
    { }

    double raycast(arma::vec3 orig, arma::vec3 dir)
    {
        orig = orig - m_center;
        arma::vec3 p;
        p(0) = 1.;
        p(1) = 2. * arma::dot(orig, dir);
        p(2) = arma::dot(orig, orig) - m_radius*m_radius;
        arma::cx_vec cx_roots = arma::roots(p);

        if (!imag(cx_roots).is_zero()) {
            return inf;
        }

        arma::vec roots = arma::real(cx_roots);
        if (roots(0) > 0. && roots(1) > 0.) {
            return arma::min(roots);
        } else if (roots(0) > 0.) {
            return roots(0);
        } else if (roots(1) > 0.) {
            return roots(1);
        } else {
            return inf;
        }
    }

    arma::vec3 normal(arma::vec3 point)
    {
        return arma::normalise(point - m_center);
    }

private:
    arma::vec3 m_center;
    double m_radius;
};

class Scene {
public:
    std::vector<Shape*> m_shapes;

    float raycast(arma::vec3 orig, arma::vec3 dir)
    {
        float min_t = inf;
        for (const auto& shape : m_shapes) {
            float t = shape->raycast(orig, dir);
            min_t = std::min(min_t, t);
        }
        return min_t;
    }
};

int main(int argc, const char *argv[])
{
    Scene scene;
    Sphere sphere1(arma::vec3{0., 0., 3.}, 1.);
    Sphere sphere2(arma::vec3{4., 0., 0.}, 1.);
    scene.m_shapes.push_back(&sphere1);
    scene.m_shapes.push_back(&sphere2);

    arma::vec3 eye {0., 0., 0.};
    arma::vec3 look {0., 0., 1.};
    arma::vec3 right {1., 0., 0.};
    arma::vec3 up {0., 1., 0.};
    int width = 640;
    int height = 480;
    double hfov = pi/2.;
    double vfov = hfov * height / width;
    arma::mat depth(height, width);

    depth.fill(4.f);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            double ty = i / (double) (height - 1);
            arma::vec3 vy = (1 - 2*ty)*tan(vfov/2)*up;
            double tx = j / (double) (width - 1);
            arma::vec3 vx = (1 - 2*tx)*tan(hfov/2)*right;
            arma::vec3 dir = arma::normalise(look + vx + vy);
            double t = scene.raycast(eye, dir);
            depth(i, j) = std::min(depth(i, j), t);
        }
        printf("%g%%\r", 100.*i / height);
        fflush(stdout);
    }
    printf("\n");

    depth.save("depth.csv", arma::csv_ascii);

    return 0;
}
