
#include <cstdio>
#include <iostream>
#include <tuple>
#include <armadillo>

using namespace arma;

const double pi = datum::pi;
const double inf = datum::inf;

namespace color {
    const vec3 white {1., 1., 1.};
    const vec3 gray {0.8, 0.8, 0.8};
    const vec3 red {1., 0., 0.};
    const vec3 green {0., 1., 0.};
    const vec3 blue {0., 0., 1.};
};

class Shape;
class Material;
std::tuple<vec3, vec3> tangents(vec3 normal);

struct Ray {
    vec3 orig;
    vec3 dir;
};

struct Hit {
    vec3 point;
    vec3 normal;
    Shape *shape;
    Material *material;
};

struct Path {
    std::vector<Ray> rays;
    std::vector<Hit> hits;
    vec3 radiance;
};

class Material {
public:
    virtual ~Material() { };
    virtual bool emissive() = 0;
    virtual vec3 brdf(vec3 normal, vec3 in, vec3 out) = 0;
    virtual vec3 emission(vec3 normal, vec3 dir) = 0;
    virtual vec3 sample(vec3 normal) = 0;
};

class DiffuseLambert : public Material {
public:
    DiffuseLambert(vec3 color)
     : m_color(color)
    { }

    ~DiffuseLambert()
    { }

    bool emissive()
    { return false; }

    vec3 brdf(vec3 normal, vec3 in, vec3 out)
    { return m_color / pi; }

    vec3 emission(vec3 normal, vec3 dir)
    { return vec3 {0., 0., 0.}; }
    
    vec3 sample(vec3 normal)
    {
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
    Emissive(vec3 emission)
     : m_emission(emission)
    { }

    ~Emissive()
    { }

    bool emissive()
    { return true; }

    vec3 brdf(vec3 normal, vec3 in, vec3 out)
    { return vec3 {0., 0., 0.}; }

    vec3 emission(vec3 normal, vec3 dir)
    { return m_emission; }

    vec3 sample(vec3 normal)
    { printf("Emissive::sample\n"); return {0., 0., 0.}; }
private:
    vec3 m_emission;
};

class Shape {
public:
    virtual bool intersect(Ray ray, double *t) = 0;
    virtual vec3 normal(vec3 point) = 0;
    virtual Material *material() = 0;
};

class Plane : public Shape {
public:
    Plane(vec3 normal, double offset, Material *material)
     : m_normal(normal), m_offset(offset), m_material(material)
    { }

    bool intersect(Ray ray, double *ot)
    {
        double h = dot(ray.orig, m_normal) - m_offset;
        double t = h / dot(ray.dir, -m_normal);
        if (t <= 0.)
            return false;
        if (ot != nullptr)
            *ot = t;
        return true;
    }

    vec3 normal(vec3 point)
    { return m_normal; }

    Material *material()
    { return m_material; }

private:
    vec3 m_normal;
    double m_offset;
    Material *m_material;
};

class Sphere : public Shape {
public:
    Sphere(vec3 center, double radius, Material *material)
      : m_center(center), m_radius(radius), m_material(material)
    { }

    bool intersect(Ray ray, double *ot)
    {
        vec3 o = ray.orig - m_center;
        vec3 p;
        p(0) = 1.;
        p(1) = 2. * dot(o, ray.dir);
        p(2) = dot(o, o) - m_radius*m_radius;
        cx_vec cx_roots;
        try {
            cx_roots = roots(p);
        } catch (const std::runtime_error& e) {
            printf("Error in roots.\n");
            return false;
        }
        if (!imag(cx_roots).is_zero()) {
            return false;
        }
        vec r_roots = real(cx_roots);
        if (r_roots(0) > 0. && r_roots(1) > 0.) {
            *ot = arma::min(r_roots);
            return true;
        } else if (r_roots(0) > 0.) {
            *ot = r_roots(0);
            return true;
        } else if (r_roots(1) > 0.) {
            *ot = r_roots(1);
            return true;
        } else {
            return false;
        }
    }

    vec3 normal(vec3 point)
    { return normalise(point - m_center); }

    Material *material()
    { return m_material; }

private:
    vec3 m_center;
    double m_radius;
    Material *m_material;
};

class Scene {
public:
    std::vector<Shape*> m_shapes;

    bool raycast(Ray ray, Hit *hit)
    {
        Shape *shape_hit;
        double tmin = inf;
        for (auto& shape : m_shapes) {
            double t;
            if (shape->intersect(ray, &t) && t < tmin) {
                shape_hit = shape;
                tmin = t;
            }
        }
        if (shape_hit == nullptr) {
            return false;
        }
        if (hit != nullptr) {
            hit->point = ray.orig + tmin*ray.dir;
            hit->normal = shape_hit->normal(hit->point);
            hit->shape = shape_hit;
            hit->material = shape_hit->material();
        }
        return true;
    }
};

class Camera {
public:
    Camera(int width,
           int height,
           double vfov = 1.3963,
           vec3 eye = vec3 {0., 0., 0.},
           vec3 forward = vec3 {0., 0., 1.},
           vec3 right = vec3 {1., 0., 0.},
           vec3 up = vec3 {0., 1., 0.})
     : m_width(width)
     , m_height(height)
     , m_vfov(vfov)
     , m_eye(eye)
     , m_forward(forward)
     , m_right(right)
     , m_up(up)
    { }

    void look_at(vec3 eye, vec3 at, vec3 up)
    {
        m_eye = eye;
        m_forward = normalise(at - eye);
        m_right = normalise(cross(m_forward, up));
        m_up = cross(m_right, m_forward);
    }

    int width() const
    { return m_width; }

    int height() const
    { return m_height; }

    double aspect() const
    { return double(m_width) / m_height; }

    Ray pix2ray(double x, double y)
    {
        double s = x / m_width;
        double t = y / m_height;
        vec3 dir = m_forward;
        dir += (2.*s - 1.) * aspect() * tan(m_vfov / 2.) * m_right;
        dir += (2.*t - 1.) * tan(m_vfov / 2.) * -m_up;
        return Ray {m_eye, normalise(dir)};
    }
private:
    int m_width;
    int m_height;
    double m_vfov;
    vec3 m_eye;
    vec3 m_forward;
    vec3 m_right;
    vec3 m_up;
};

std::tuple<vec3, vec3> tangents(vec3 normal)
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

bool pathtrace(Scene& scene, Ray ray, Path *path, double absorb)
{
    *path = Path { };
    for (;;) {
        if (randu() < absorb) {
            return false;
        }
        Hit hit;
        if (!scene.raycast(ray, &hit)) {
            return false;
        }
        path->rays.push_back(ray);
        if (hit.material->emissive()) {
            path->radiance = hit.material->emission(hit.normal, -ray.dir);
            return true;
        }
        path->hits.push_back(hit);
        vec3 dir = hit.material->sample(hit.normal);
        vec3 orig = hit.point + 1e-3*dir;
        ray = Ray {orig, dir};
    }
}

vec3 forward(Path& path)
{
    vec3 radiance = path.radiance;
    for (int i = path.hits.size()-1; i >= 0; --i) {
        Hit hit = path.hits[i];
        vec3 in = path.rays[i].dir;
        vec3 out = path.rays[i+1].dir;
        vec3 brdf = hit.material->brdf(hit.normal, -in, out);
        radiance = brdf % radiance;
    }
    return radiance;
}

void serialize(FILE *fp, cube& img)
{
    fprintf(fp, "[\n");
    for (int i = 0; i < img.n_rows; ++i) {
        fprintf(fp, "  [\n");
        for (int j = 0; j < img.n_cols; ++j) {
            fprintf(fp, "    [");
            for (int k = 0; k < img.n_slices; ++k) {
                fprintf(fp, "%f", img(i, j, k));
                if (k != img.n_slices - 1) {
                    fprintf(fp, ", ");
                }
            }
            fprintf(fp, "]");
            if (j != img.n_cols - 1) {
                fprintf(fp, ", ");
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "  ]");
        if (i != img.n_rows - 1) {
            fprintf(fp, ", ");
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "]\n");
}

int main(int argc, const char *argv[])
{
    Scene scene;
    Material *red = new DiffuseLambert(color::red);
    Material *blue = new DiffuseLambert(color::blue);
    Material *white = new DiffuseLambert(color::white);
    Material *emissive = new Emissive(color::white);
    Sphere sphere1(vec3{0., 0., 3.}, 1., red);
    Sphere sphere2(vec3{1., 1., 4.5}, 1., blue);
    Sphere sphere3(vec3{0., 3., 3.}, 1.0, emissive);
    Plane plane1(vec3{1., 0., 0.}, -3., white);
    Plane plane2(vec3{-1., 0., 0.1}, -3., white);
    Plane plane3(vec3{0., 1., 0.}, -3., white);
    Plane plane4(vec3{0., -1., 0.}, -3., white);
    Plane plane5(vec3{0., 0., -1.}, -6., white);
    scene.m_shapes.push_back(&sphere1);
    scene.m_shapes.push_back(&sphere2);
    scene.m_shapes.push_back(&sphere3);
    scene.m_shapes.push_back(&plane1);
    scene.m_shapes.push_back(&plane2);
    scene.m_shapes.push_back(&plane3);
    scene.m_shapes.push_back(&plane4);
    scene.m_shapes.push_back(&plane5);

    Camera cam(640, 480);
    cube img(480, 640, 3);
    const int n_samples = 100;
    const double absorb = 1./3.; // 2 bounces on avg.

    for (int y = 0; y < cam.height(); ++y) {
        for (int x = 0; x < cam.width(); ++x) {
            Ray ray = cam.pix2ray(x, y);
            vec3 color {0., 0., 0.};
            for (int k = 0; k < n_samples; ++k) {
                Path path;
                if (!pathtrace(scene, ray, &path, absorb)) {
                    continue;
                }
                color += forward(path) / (absorb * n_samples);
            }
            img.tube(y, x) = color;
        }
        printf("% 5.2f%%\r", 100. * (y+1) / cam.height());
        fflush(stdout);
    }
    printf("\n");

    FILE *fp = fopen("out.json", "w");
    serialize(fp, img);
    fclose(fp);

    return 0;
}
