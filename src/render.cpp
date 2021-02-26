#include <stdio.h>
#include "args.hpp"
#include "bxdf.hpp"
#include "camera.hpp"
#include "material.hpp"
#include "pathtracer.hpp"
#include "scene.hpp"
#include "shape.hpp"
#include "vector.hpp"
#include "write.hpp"

using namespace drt;

int main(int argc, const char *argv[])
{
    Args args;
    if (!parse_args(argc, argv, &args)) {
        return EXIT_FAILURE;
    }

    // Configure scene parameters
    Var3 red_var(red, true);
    Var3 green_var(green, true);
    Var3 white_var(white, true);
    Var3 emission_var(white, true);

    // Configure scene materials
    DiffuseBxDF red_bxdf(red_var);
    DiffuseBxDF green_bxdf(green_var);
    DiffuseBxDF white_bxdf(white_var);
    DiffuseBxDF black_bxdf(black);

    Material red_mat {&red_bxdf, black};
    Material green_mat {&green_bxdf, black};
    Material white_mat {&white_bxdf, black};
    Material emissive_mat {&black_bxdf, emission_var};

    // Configure scene shapes
    Sphere sphere1(Vec3 {0., 0., 3.}, 1.);
    Sphere sphere2(Vec3 {-1., 1., 4.5}, 1.);
    Sphere sphere3(Vec3 {0., 3., 3.}, 1.);
    Plane plane1(Vec3 {-1., 0., 0.}, -3.);
    Plane plane2(Vec3 {1., 0., 0.1}, -3.);
    Plane plane3(Vec3 {0., 1., 0.}, -3.);
    Plane plane4(Vec3 {0., -1., 0.}, -3.);
    Plane plane5(Vec3 {0., 0., -1.}, -6.);

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
    std::size_t width = 640;
    std::size_t height = 480;
    Camera cam(width, height);
    Vec3 *img = new Vec3[width * height];

    // Configure path tracer sampling
    Pathtracer tracer(args.absorb_prob, args.min_bounces);

    // Render test scene
    for (int y = 0; y < cam.height(); ++y) {
        for (int x = 0; x < cam.width(); ++x) {
            Vec3 orig;
            Vec3 dir;
            cam.pix2ray(x, y, orig, dir);
            Vec3 radiance(0);
            for (int k = 0; k < args.samples; ++k) {
                Var3 rad = tracer.trace(scene, orig, dir);
                radiance += rad.detach() / double(args.samples);
                rad.backward(Vec3(1) / double(args.samples));
            }
            // img[y*width + x] = radiance;
            img[y*width + x]  = red_var.grad();
            red_var.grad() = Vec3(0);
            // img[i*width + j] = emission.detach();
            // emission_var.grad() = vec3(fill::zeros);
        }
        printf("% 5.2f%%\r", 100. * (y+1) / cam.height());
        fflush(stdout);
    }
    printf("\n");

    // Write radiance to file
    write_exr(args.output.c_str(), img, width, height);

    return 0;
}
