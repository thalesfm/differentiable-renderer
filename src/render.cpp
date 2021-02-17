#include <stdio.h>
#include <tuple>
#include <armadillo>
#include "args.hpp"
#include "autograd.hpp"
#include "brdf.hpp"
#include "camera.hpp"
#include "common.hpp"
#include "material.hpp"
#include "pathtracer.hpp"
#include "scene.hpp"
#include "shape.hpp"
#include "write.hpp"

using namespace arma;
using namespace drt;

int main(int argc, const char *argv[])
{
    Args args;
    if (!parse_args(argc, argv, &args)) {
        return EXIT_FAILURE;
    }

    // Configure scene parameters
    Variable<rgb> red_var(red);
    Variable<rgb> green_var(green);
    Variable<rgb> white_var(white);
    Variable<rgb> emission_var(white);
    Constant<rgb> no_emission_const(black);

    // Configure scene materials
    DiffuseBRDF red_brdf(red_var);
    DiffuseBRDF green_brdf(green_var);
    DiffuseBRDF white_brdf(white_var);
    BlackBRDF black_brdf;
    Material red_mat {&red_brdf, &no_emission_const};
    Material green_mat {&green_brdf, &no_emission_const};
    Material white_mat {&white_brdf, &no_emission_const};
    Material emissive_mat {&black_brdf, &emission_var};

    // Configure scene shapes
    Sphere sphere1(vec3{0., 0., 3.}, 1.);
    Sphere sphere2(vec3{-1., 1., 4.5}, 1.);
    Sphere sphere3(vec3{0., 3., 3.}, 1.);
    Plane plane1(vec3{-1., 0., 0.}, -3.);
    Plane plane2(vec3{1., 0., 0.1}, -3.);
    Plane plane3(vec3{0., 1., 0.}, -3.);
    Plane plane4(vec3{0., -1., 0.}, -3.);
    Plane plane5(vec3{0., 0., -1.}, -6.);

    // Build test scene
    Scene scene;
    scene.add(sphere1, white_mat);
    scene.add(sphere2, white_mat);
    scene.add(sphere3, emissive_mat);
    scene.add(plane1, red_mat);
    scene.add(plane2, green_mat);
    scene.add(plane3, white_mat);
    scene.add(plane4, white_mat);
    scene.add(plane5, white_mat);

    // Configure camera position and resolution
    Camera cam(640, 480);
    cube img(480, 640, 3);

    // Configure path tracer sampling
    Pathtracer tracer(args.absorb_prob, args.min_bounces);

    // Render test scene
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

    // Write radiance to file
    write_exr(args.output.c_str(), img);

    return 0;
}
