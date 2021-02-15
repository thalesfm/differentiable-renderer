#include <stdio.h>
#include <tuple>
#include <armadillo>
#include <tclap/CmdLine.h>
#include "args.hpp"

using namespace arma;
using namespace drt;
using namespace drt::args;

const double pi = datum::pi;
const double inf = datum::inf;

namespace color {
    const vec3 white {1., 1., 1.};
    const vec3 gray {0.8, 0.8, 0.8};
    const vec3 red {1., 0., 0.};
    const vec3 green {0., 1., 0.};
    const vec3 blue {0., 0., 1.};
};

struct Ray {
    vec3 orig;
    vec3 dir;
};

struct Options {
    double absorb;
};

class Material {
public:
    virtual ~Material() { };

    virtual vec3 sample(vec3 normal) = 0;

    virtual vec3 brdf_pdf(vec3 normal, vec3 dir_in, vec3 dir_out) = 0;

    virtual vec3 emission(vec3 normal, vec3 dir_in) = 0;

    virtual bool pure_emissive() = 0;

    vec3& grad()
    {
        return m_grad;
    }
private:
    vec3 m_grad;
};

std::tuple<vec3, vec3> tangents(vec3 normal);

class DiffuseLambert : public Material {
public:
    DiffuseLambert(vec3 color)
      : m_color(color)
    { }

    ~DiffuseLambert()
    { }

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

    vec3 brdf_pdf(vec3 normal, vec3 dir_in, vec3 dir_out)
    {
        return m_color / pi;
    }

    vec3 emission(vec3 normal, vec3 dir_in)
    {
        return vec3(fill::zeros);
    }
    
    bool pure_emissive()
    {
        return false;
    }

private:
    vec3 m_color;
    vec3 m_color_grad;
};

class Emissive : public Material {
public:
    Emissive(vec3 emission)
     : m_emission(emission)
    { }

    ~Emissive()
    { }

    bool pure_emissive()
    {
        return true;
    }

    vec3 brdf_pdf(vec3 normal, vec3 in, vec3 out)
    {
        return vec3(fill::zeros);
    }

    vec3 emission(vec3 normal, vec3 dir)
    {
        return m_emission;
    }

    vec3 sample(vec3 normal)
    {
        throw std::runtime_error("Attempt to sample from emissive material");
    }
private:
    vec3 m_emission;
};

struct Point {
    vec3 pos;
    vec3 normal;
    Material *material;
};

class Surface {
public:
    virtual bool intersect(Ray ray, double *t) = 0;
    virtual vec3 normal(vec3 point) = 0;
    virtual Material *material() = 0;
};

class Plane : public Surface {
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

class Sphere : public Surface {
public:
    Sphere(vec3 center, double radius, Material *material)
      : m_center(center), m_radius(radius), m_material(material)
    { }

    bool intersect(Ray ray, double *t)
    {
        vec3 o = ray.orig - m_center;
        double a = 1.;
        double b = 2. * dot(o, ray.dir);
        double c = dot(o, o) - m_radius*m_radius;
        double d = b*b - 4.*a*c;
        if (d < 0.) {
            return false;
        }
        double t1 = (-b - sqrt(d)) / (2. * a);
        double t2 = (-b + sqrt(d)) / (2. * a);
        if (t1 > 0. && t2 > 0.) {
            *t = std::min(t1, t2);
            return true;
        } else if (t1 > 0.) {
            *t = t1;
            return true;
        } else if (t2 > 0.) {
            *t = t2;
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
    std::vector<Surface*> m_surfaces;

    bool raycast(Ray ray, Point *point)
    {
        Surface *surface_hit = nullptr;
        double tmin = inf;
        for (auto& surface : m_surfaces) {
            double t;
            if (surface->intersect(ray, &t) && t < tmin) {
                surface_hit = surface;
                tmin = t;
            }
        }
        if (surface_hit == nullptr) {
            return false;
        }
        if (point != nullptr) {
            point->pos = ray.orig + tmin*ray.dir;
            point->normal = surface_hit->normal(point->pos);
            point->material = surface_hit->material();
        }
        return true;
    }
};

class Path;

Path *path_trace(Scene& scene, Args& args, Ray ray, int depth = 0);
vec3 brdf_pdf_sample(Point point, vec3 dir_in, vec3 *dir_out);
vec3 emission(Point point, vec3 dir_in);
bool pure_emissive(Point point);

class Path {
public:
    virtual ~Path() { }
    virtual vec3 forward() = 0;
    virtual void backward(vec3 weight) = 0;
};

class Miss : public Path {
public:
    Miss(Scene& scene, Args& args, vec3 dir_in)
      : m_scene(scene)
      , m_dir_in(dir_in)
    { }

    vec3 forward()
    {
        return vec3(fill::zeros);
    }

    void backward(vec3 weight)
    { }
private:
    Scene& m_scene;
    vec3 m_dir_in;
};

class Scatter : public Path {
public:
    Scatter(Scene& scene, Args& args, Point point, vec3 dir_in, int depth)
      : m_scene(scene)
      , m_point(point)
      , m_dir_in(dir_in)
    {
        m_brdf_pdf_value = brdf_pdf_sample(m_point, m_dir_in, &m_dir_out);
        m_next_path = path_trace(m_scene, args, Ray {point.pos + 1e-3*m_dir_out, m_dir_out}, depth);
    }

    ~Scatter()
    {
        delete m_next_path;
    }

    vec3 forward()
    {
        return m_brdf_pdf_value % m_next_path->forward();
    }

    void backward(vec3 weight)
    {
        // TODO: Store m_next_path->forward()
        // TODO: Consider absorbtion prob.
        m_point.material->grad() += m_next_path->forward() % weight;
        m_next_path->backward(m_brdf_pdf_value % weight);
    }
private:
    Scene& m_scene;
    Point m_point;
    vec3 m_dir_in;
    vec3 m_dir_out;
    vec3 m_brdf_pdf_value;
    Path *m_next_path;
};

class Emission : public Path {
public:
    Emission(Scene& scene, Args& args, Point point, vec3 dir_in)
      : m_point(point)
      , m_dir_in(dir_in)
    { }

    vec3 forward()
    {
        return emission(m_point, m_dir_in);
    }

    void backward(vec3 weight)
    {
        m_point.material->grad() += weight;
    }
private:
    Point m_point;
    vec3 m_dir_in;
};

Path *path_trace(Scene& scene, Args& args, Ray ray, int depth)
{
    if (depth >= args.min_bounces && randu() < args.absorb_prob) {
        return new Miss(scene, args, ray.dir);
    }
    Point point;
    if (!scene.raycast(ray, &point)) {
        return new Miss(scene, args, ray.dir);
    }
    if (pure_emissive(point)) {
        return new Emission(scene, args, point, ray.dir);
    }
    return new Scatter(scene, args, point, ray.dir, depth+1);
}

vec3 brdf_pdf_sample(Point point, vec3 dir_in, vec3 *dir_out)
{
    Material *material = point.material;
    *dir_out = material->sample(point.normal);
    return material->brdf_pdf(point.normal, dir_in, *dir_out);
}

vec3 emission(Point point, vec3 dir_in)
{
    return point.material->emission(point.normal, dir_in);
}

bool pure_emissive(Point point)
{
    return point.material->pure_emissive();
}

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
    Args args;
    if (!args::parse(argc, argv, &args)) {
        return EXIT_FAILURE;
    }

    // Build test scene
    Scene scene;
    Material *red = new DiffuseLambert(color::red);
    Material *green = new DiffuseLambert(color::green);
    Material *white = new DiffuseLambert(color::white);
    Material *emissive = new Emissive(color::white);
    Sphere sphere1(vec3{0., 0., 3.}, 1., white);
    Sphere sphere2(vec3{1., 1., 4.5}, 1., white);
    Sphere sphere3(vec3{0., 3., 3.}, 1.0, emissive);
    Plane plane1(vec3{1., 0., 0.}, -3., red);
    Plane plane2(vec3{-1., 0., 0.1}, -3., green);
    Plane plane3(vec3{0., 1., 0.}, -3., white);
    Plane plane4(vec3{0., -1., 0.}, -3., white);
    Plane plane5(vec3{0., 0., -1.}, -6., white);
    scene.m_surfaces.push_back(&sphere1);
    scene.m_surfaces.push_back(&sphere2);
    scene.m_surfaces.push_back(&sphere3);
    scene.m_surfaces.push_back(&plane1);
    scene.m_surfaces.push_back(&plane2);
    scene.m_surfaces.push_back(&plane3);
    scene.m_surfaces.push_back(&plane4);
    scene.m_surfaces.push_back(&plane5);

    Camera cam(640, 480);
    cube img(480, 640, 3);

    for (int y = 0; y < cam.height(); ++y) {
        for (int x = 0; x < cam.width(); ++x) {
            Ray ray = cam.pix2ray(x, y);
            vec3 color {0., 0., 0.};
            for (int k = 0; k < args.samples; ++k) {
                Path *path = path_trace(scene, args, ray);
                color += path->forward() / args.samples;
                path->backward(vec3{1., 1., 1.} / args.samples);
                delete path;
            }
            img.tube(y, x) = color;
            // img.tube(y, x) = red->grad();
            // red->grad() = vec3(fill::zeros);
        }
        printf("% 5.2f%%\r", 100. * (y+1) / cam.height());
        fflush(stdout);
    }
    printf("\n");

    FILE *fp = fopen(args.output.c_str(), "w");
    serialize(fp, img);
    fclose(fp);

    return 0;
}
