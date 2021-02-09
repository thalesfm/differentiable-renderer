#include <cmath>
#include <cstdio>
#include <iostream>
#include <tuple>
#include <armadillo>

using namespace arma;

const double pi = datum::pi;
const double inf = datum::inf;

std::tuple<vec3, vec3> tangents(vec3 normal)
{
    vec3 e1 {1., 0., 0.};
    vec3 e2 {0., 1., 0.};

    vec3 tangent;
    if (dot(e1, normal) < dot(e2, normal)) {
        tangent = normalise(e1 - normal*dot(e1, normal));
    } else {
        tangent = normalise(e2 - normal*dot(e2, normal));
    }

    vec3 bitangent = normalise(cross(normal, tangent));
    return std::make_tuple(tangent, bitangent);
}

namespace color {
    const vec3 white {1., 1., 1.};
    const vec3 gray {0.8, 0.8, 0.8};
    const vec3 red {1., 0., 0.};
    const vec3 green {0., 1., 0.};
    const vec3 blue {0., 0., 1.};
};

struct RayHit;

class Material {
public:
    virtual ~Material() { };

    virtual vec3 brdf(vec3 normal, vec3 in, vec3 out)
    { return {0., 0., 0.}; }

    virtual vec3 emission(vec3 normal, vec3 in, vec3 out)
    { return {0., 0., 0.}; }

    virtual vec3 sample(vec3 normal)
    { printf("Material::sample\n"); return {0., 0., 0.}; }
};

// vec3 polar2cartesian(...);

class DiffuseLambert : public Material {
public:
    DiffuseLambert(vec3 color) : m_color(color) { }

    ~DiffuseLambert() { }

    vec3 brdf(vec3 normal, vec3 in, vec3 out)
    { return m_color / pi; }
    
    vec3 sample(vec3 normal)
    {
        // Double check the math for this
        double theta = acos(sqrt(1 - randu()));
        double phi = 2*pi*randu();

        vec3 tangent, bitangent;
        std::tie(tangent, bitangent) = tangents(normal);

        double dir_t = cos(phi) * cos(theta);
        double dir_b = sin(phi) * cos(theta);
        double dir_n = sin(theta);
        return dir_t*tangent + dir_b*bitangent + dir_n*normal;
    }
private:
    vec3 m_color;
};

class Emissive : public Material {
public:
    Emissive(vec3 emission) : m_emission(emission) { }

    ~Emissive() { }

    vec3 emission(vec3 normal, vec3 in, vec3 out)
    { return m_emission; }

    vec3 sample(vec3 normal)
    { printf("Emissive::sample\n"); return {0., 0., 0.}; }
private:
    vec3 m_emission;
};

class Shape {
public:
    virtual double raycast(vec3 orig, vec3 dir) = 0;
    virtual vec3 normal(vec3 point) = 0;
    virtual Material *material() = 0;
};

class Sphere : public Shape {
public:
    Sphere(vec3 center, double radius, Material *material)
      : m_center(center), m_radius(radius), m_material(material)
    { }

    double raycast(vec3 orig, vec3 dir)
    {
        orig = orig - m_center;
        vec3 p;
        p(0) = 1.;
        p(1) = 2. * dot(orig, dir);
        p(2) = dot(orig, orig) - m_radius*m_radius;
        cx_vec cx_roots = roots(p);

        if (!imag(cx_roots).is_zero()) {
            return inf;
        }

        vec r_roots = real(cx_roots);
        if (r_roots(0) > 0. && r_roots(1) > 0.) {
            return arma::min(r_roots);
        } else if (r_roots(0) > 0.) {
            return r_roots(0);
        } else if (r_roots(1) > 0.) {
            return r_roots(1);
        } else {
            return inf;
        }
    }

    vec3 normal(vec3 point)
    {
        return normalise(point - m_center);
    }

    Material *material()
    {
        return m_material;
    }

private:
    vec3 m_center;
    double m_radius;
    Material *m_material;
};

class Scene {
public:
    std::vector<Shape*> m_shapes;

    float raycast(vec3 orig, vec3 dir, RayHit *hit);
};

struct RayHit {
    vec3 point;
    vec3 normal;
    Shape *shape;
    Material *material;
};

struct RayBounce {
    vec3 in;
    vec3 out;
    vec3 normal;
    Material *material;
};

float Scene::raycast(vec3 orig, vec3 dir, RayHit *hit)
{
    float closest_t = inf;
    for (const auto& shape : m_shapes) {
        float t = shape->raycast(orig, dir);
        if (t >= closest_t) {
            continue;
        }
        if (hit != nullptr) {
            closest_t = t;
            hit->point = orig + t*dir;
            hit->normal = shape->normal(hit->point);
            hit->shape = shape;
            hit->material = shape->material();
        }
    }
    return closest_t;
}

template <typename OutputIt>
bool pathtrace(Scene& scene, vec3 orig, vec3 dir, OutputIt path, vec3 *emission) {
    // For num. bounces
    for (int i = 0; i < 3; ++i) {
        RayHit hit;
        double t = scene.raycast(orig, dir, &hit);
        if (std::isinf(t)) {
            return false;
        }
        vec3 e = hit.material->emission(hit.normal, -dir, vec3 {0., 0., 0.});
        if (e.is_zero()) {
            vec3 out = hit.material->sample(hit.normal);
            RayBounce bounce;
            bounce.in = dir;
            bounce.out = out;
            bounce.normal = hit.normal;
            bounce.material = hit.material;
            *path++ = bounce;
            orig = orig + (t + 1e-3)*dir;
            dir = out;
        } else {
            *emission = e;
            return true;
        }
    }
    return false;
}

vec3 forward(std::vector<RayBounce>& path, vec3 radiance)
{
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        RayBounce bounce = *it;
        vec3 brdf = bounce.material->brdf(bounce.normal, -bounce.in, bounce.out);
        radiance = brdf%radiance;
    }
    return radiance;
}

int main(int argc, const char *argv[])
{
    Scene scene;
    Material *red = new DiffuseLambert(color::red);
    Material *blue = new DiffuseLambert(color::blue);
    Material *white = new DiffuseLambert(color::white);
    Material *emissive = new Emissive(color::white);
    Sphere sphere1(vec3{0., 0., 3.}, 1., red);
    Sphere sphere2(vec3{1., 1., 5.}, 1., blue);
    Sphere sphere3(vec3{-103., 0., 3.}, 100., white);
    Sphere sphere4(vec3{103., 0., 3.}, 100., white);
    Sphere sphere5(vec3{0., -103., 3.}, 100., white);
    Sphere sphere6(vec3{0., 103., 3.}, 100., white);
    Sphere sphere7(vec3{0., 3., 3.}, 1.0, emissive);
    scene.m_shapes.push_back(&sphere1);
    scene.m_shapes.push_back(&sphere2);
    scene.m_shapes.push_back(&sphere3);
    scene.m_shapes.push_back(&sphere4);
    scene.m_shapes.push_back(&sphere5);
    scene.m_shapes.push_back(&sphere6);
    scene.m_shapes.push_back(&sphere7);

    vec3 eye {0., 0., 0.};
    vec3 look {0., 0., 1.};
    vec3 right {1., 0., 0.};
    vec3 up {0., 1., 0.};
    int width = 640;
    int height = 480;
    double hfov = pi/2.;
    double vfov = hfov * height / width;
    cube image(height, width, 3);

    for (int i = 0; i < height; ++i) {
        double t = i / (double) (height - 1);
        vec3 dy = (1 - 2*t)*tan(vfov/2)*up;

        for (int j = 0; j < width; ++j) {
            double t = j / (double) (width - 1);
            vec3 dx = -(1 - 2*t)*tan(hfov/2)*right;
            vec3 dir = normalise(look + dx + dy);
            vec3 color {0., 0., 0.};

            for (int k = 0; k < 4; ++k) {
                std::vector<RayBounce> path;
                vec3 radiance;
                bool ret = pathtrace(scene, eye, dir, std::back_inserter(path), &radiance);
                if (ret) {
                    color += forward(path, radiance) / 4;
                }
            }

            image.tube(i, j) = color;
        }
        printf("%g%%\r", 100.*(i+1)/height);
        fflush(stdout);
    }
    printf("\n");

    image.slice(0).save("out_r.csv", csv_ascii);
    image.slice(1).save("out_g.csv", csv_ascii);
    image.slice(2).save("out_b.csv", csv_ascii);

    return 0;
}
