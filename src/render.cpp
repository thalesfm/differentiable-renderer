#include <stdio.h>
#include "args.hpp"
#include "bxdf.hpp"
#include "camera.hpp"
#include "emitter.hpp"
#include "integrate.hpp"
#include "pathtracer.hpp"
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
    Var3 red_var(0.5 * red, true);
    Var3 green_var(0.5 * green, true);
    Var3 white_var(0.5 * white, true);
    Var3 emission_var(white, true);

    // Configure scene materials
    auto red_bxdf = std::make_shared<DiffuseBxDF>(red_var);
    auto green_bxdf = std::make_shared<DiffuseBxDF>(green_var);
    auto white_bxdf = std::make_shared<DiffuseBxDF>(white_var);
    auto emitter = std::make_shared<AreaEmitter>(white);

    // Configure scene shapes
    Sphere sphere1(Vec3{0., 0., 3.}, 1., white_bxdf);
    Sphere sphere2(Vec3{-1., 1., 4.5}, 1., white_bxdf);
    Sphere sphere3(Vec3{0., 3., 3.}, 1., nullptr, emitter);
    Plane plane1(Vec3{-1., 0., 0.}, -3., red_bxdf);
    Plane plane2(Vec3{1., 0., 0.1}, -3., green_bxdf);
    Plane plane3(Vec3{0., 1., 0.}, -3., white_bxdf);
    Plane plane4(Vec3{0., -1., 0.}, -3., white_bxdf);
    Plane plane5(Vec3{0., 0., -1.}, -6., white_bxdf);

    // Build test scene
    Scene scene;
    scene.push_back(&sphere1);
    scene.push_back(&sphere2);
    scene.push_back(&sphere3);
    scene.push_back(&plane1);
    scene.push_back(&plane2);
    scene.push_back(&plane3);
    scene.push_back(&plane4);
    scene.push_back(&plane5);

    // Configure camera position and resolution
    std::size_t width = args.width;
    std::size_t height = args.height;
    Camera cam(width, height);
    Vec3 *img = new Vec3[width * height];

    // Configure path tracer sampling
    Pathtracer tracer(args.absorb_prob, args.min_bounces);

    // Render test scene
    for (int y = 0; y < cam.height(); ++y) {
        for (int x = 0; x < cam.width(); ++x) {
            // Vec3 pixel_radiance(0);
            Vec3 red_grad(0);
            for (std::size_t i = 0; i < args.samples; ++i) {
                Vec3 dir = cam.sample(x, y);
                Var3 radiance = tracer.trace(scene, cam.eye(), dir);
                // pixel_radiance += radiance.detach();
                red_var.grad() = Vec3(0);
                radiance.backward(Vec3(1));
                red_grad += red_var.grad();
            }
            // img[y*width + x] = pixel_radiance / args.samples;
            img[y*width + x] = red_grad / args.samples;
        }
        printf("% 5.2f%%\r", 100. * (y+1) / cam.height());
        fflush(stdout);
    }
    printf("\n");

    // Write radiance to file
    write_exr(args.output.c_str(), img, width, height);

    return 0;
}
