#include <stdio.h>
#include <tuple>
#include <armadillo>
#include "args.hpp"
#include "autograd.hpp"
#include "brdf.hpp"
#include "common.hpp"
#include "material.hpp"
#include "pathtracer.hpp"
#include "scene.hpp"
#include "shape.hpp"
#include "write.hpp"

using namespace arma;
using namespace drt;

class Camera {
public:
    Camera(int width,
           int height,
           double vfov = 1.3963,
           vec3 eye = vec3 {0., 0., 0.},
           vec3 forward = vec3 {0., 0., 1.},
           vec3 right = vec3 {-1., 0., 0.},
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

    void pix2ray(double x, double y, vec3 &orig, vec3 &dir)
    {
        orig = m_eye;
        double s = x / m_width;
        double t = y / m_height;
        dir = m_forward;
        dir += (2.*s - 1.) * aspect() * tan(m_vfov / 2.) * m_right;
        dir += (2.*t - 1.) * tan(m_vfov / 2.) * -m_up;
        dir = normalise(dir);
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

int main(int argc, const char *argv[])
{
    Args args;
    if (!parse_args(argc, argv, &args)) {
        return EXIT_FAILURE;
    }

    // Build test scene
    Scene scene;
    Variable<rgb> red_var(red);
    Variable<rgb> green_var(green);
    Variable<rgb> white_var(white);
    Variable<rgb> emission_var(white);
    Constant<rgb> no_emission_const(black);
    DiffuseBRDF red_brdf(red_var);
    DiffuseBRDF green_brdf(green_var);
    DiffuseBRDF white_brdf(white_var);
    BlackBRDF black_brdf;
    Material red_mat {&red_brdf, &no_emission_const};
    Material green_mat {&green_brdf, &no_emission_const};
    Material white_mat {&white_brdf, &no_emission_const};
    Material emissive_mat {&black_brdf, &emission_var};
    Sphere sphere1(vec3{0., 0., 3.}, 1.);
    Sphere sphere2(vec3{-1., 1., 4.5}, 1.);
    Sphere sphere3(vec3{0., 3., 3.}, 1.);
    Plane plane1(vec3{-1., 0., 0.}, -3.);
    Plane plane2(vec3{1., 0., 0.1}, -3.);
    Plane plane3(vec3{0., 1., 0.}, -3.);
    Plane plane4(vec3{0., -1., 0.}, -3.);
    Plane plane5(vec3{0., 0., -1.}, -6.);

    scene.add(sphere1, white_mat);
    scene.add(sphere2, white_mat);
    scene.add(sphere3, emissive_mat);
    scene.add(plane1, red_mat);
    scene.add(plane2, green_mat);
    scene.add(plane3, white_mat);
    scene.add(plane4, white_mat);
    scene.add(plane5, white_mat);

    Camera cam(640, 480);
    cube img(480, 640, 3);

    Pathtracer tracer(args.absorb_prob, args.min_bounces);

    for (int y = 0; y < cam.height(); ++y) {
        for (int x = 0; x < cam.width(); ++x) {
            vec3 orig;
            vec3 dir;
            cam.pix2ray(x, y, orig, dir);
            rgb radiance {0., 0., 0.};
            for (int k = 0; k < args.samples; ++k) {
                Autograd<rgb> *rad = tracer.trace(scene, orig, dir);
                radiance += rad->value() / args.samples;
                rad->backward(vec3(fill::ones) / args.samples);
                delete rad;
            }
            // img.tube(y, x) = radiance;
            // img.tube(y, x) = red_var.grad();
            // red_var.grad() = vec3(fill::zeros);
            img.tube(y, x) = emission_var.grad();
            emission_var.grad() = vec3(fill::zeros);
        }
        printf("% 5.2f%%\r", 100. * (y+1) / cam.height());
        fflush(stdout);
    }
    printf("\n");

    write_exr(args.output.c_str(), img);

    return 0;
}
