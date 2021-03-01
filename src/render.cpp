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

    using T = double;

    // Configure scene parameters
    Vector<T, 3, true> red(Vector<T, 3>{0.5, 0, 0}, true);
    Vector<T, 3, true> green(Vector<T, 3>{0, 0.5, 0}, true);
    Vector<T, 3, true> white(Vector<T, 3>{0.5, 0.5, 0.5}, true);
    Vector<T, 3, true> emission(Vector<T, 3>(1), true);

    // Configure scene materials
    auto diffuse_red = std::make_shared<DiffuseBxDF<T>>(red);
    auto diffuse_green = std::make_shared<DiffuseBxDF<T>>(green);
    auto diffuse_white = std::make_shared<DiffuseBxDF<T>>(white);
    auto emitter = std::make_shared<AreaEmitter<T>>(emission);

    // Configure scene shapes
    Sphere<T> sphere1(Vector<T, 3>{0., 0., 3.}, 1., diffuse_white);
    Sphere<T> sphere2(Vector<T, 3>{-1., 1., 4.5}, 1., diffuse_white);
    Sphere<T> sphere3(Vector<T, 3>{0., 3., 3.}, 1., nullptr, emitter);
    Plane<T> plane1(Vector<T, 3>{-1., 0., 0.}, -3., diffuse_red);
    Plane<T> plane2(Vector<T, 3>{1., 0., 0.1}, -3., diffuse_green);
    Plane<T> plane3(Vector<T, 3>{0., 1., 0.}, -3., diffuse_white);
    Plane<T> plane4(Vector<T, 3>{0., -1., 0.}, -3., diffuse_white);
    Plane<T> plane5(Vector<T, 3>{0., 0., -1.}, -6., diffuse_white);

    // Build test scene
    Scene<T> scene;
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
    Camera<T> cam(width, height);
    Vector<T, 3> *img = new Vector<T, 3>[width * height];

    // Configure path tracer sampling
    Pathtracer<T> tracer(args.absorb_prob, args.min_bounces);

    // Render test scene
    for (int y = 0; y < cam.height(); ++y) {
        for (int x = 0; x < cam.width(); ++x) {
            Vector<T, 3> pixel_radiance(0);
            // Vec3 red_grad(0);
            for (std::size_t i = 0; i < args.samples; ++i) {
                Vector<T, 3> dir = cam.sample(x, y);
                Vector<T, 3, true> radiance = tracer.trace(scene, cam.eye(), dir);
                pixel_radiance += radiance.detach();
                // red.grad() = Vec3(0);
                // radiance.backward(Vec3(1));
                // red_grad += red.grad();
            }
            img[y*width + x] = pixel_radiance / args.samples;
            // img[y*width + x] = red_grad / args.samples;
        }
        printf("% 5.2f%%\r", 100. * (y+1) / cam.height());
        fflush(stdout);
    }
    printf("\n");

    // Write radiance to file
    write_exr(args.output.c_str(), img, width, height);

    return 0;
}
