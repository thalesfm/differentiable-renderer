#include <stdio.h>
#include "drt/bxdf.hpp"
#include "drt/camera.hpp"
#include "drt/dual.hpp"
#include "drt/emitter.hpp"
#include "drt/integrate.hpp"
#include "drt/pathtracer.hpp"
#include "drt/shape.hpp"
#include "drt/vector.hpp"
#include "args.hpp"
#include "write.hpp"

using namespace drt;

int main(int argc, const char *argv[])
{
    Args args;
    if (!parse_args(argc, argv, &args)) {
        return EXIT_FAILURE;
    }

    using T = double;
    // using T = Dual<double>;

    // Configure scene parameters
    Vector<T, 3, true> red(Vector<T, 3>{0.5, 0, 0}, true);
    Vector<T, 3, true> green(Vector<T, 3>{0, 0.5, 0}, true);
    Vector<T, 3, true> white(Vector<T, 3>{0.5, 0.5, 0.5}, true);
    Vector<T, 3, true> emission(Vector<T, 3>(1), true);

    // Configure scene materials
    auto diffuse_red = std::make_shared<DiffuseBxDF<T>>(red);
    auto diffuse_green = std::make_shared<DiffuseBxDF<T>>(green);
    auto diffuse_white = std::make_shared<DiffuseBxDF<T>>(white);
    auto specular_white = std::make_shared<SpecularBxDF<T>>(white, 30);
    auto emitter = std::make_shared<AreaEmitter<T>>(emission);

    // Configure scene shapes
    Sphere<T> sphere_front(Vector<T, 3>{0., 0., 3.}, 1., diffuse_white);
    Sphere<T> sphere_back(Vector<T, 3>{-1., 1., 4.5}, 1., diffuse_white);
    Plane<T> left_plane(Vector<T, 3>{-1., 0., 0.}, -3., diffuse_red);
    Plane<T> right_plane(Vector<T, 3>{1., 0., 0.1}, -3., diffuse_green);
    Plane<T> back_plane(Vector<T, 3>{0., 0., -1.}, -6., diffuse_white);
    Plane<T> front_plane(Vector<T, 3>{0, 0, 1}, 0, diffuse_white);
    Plane<T> ground_plane(Vector<T, 3>{0., 1., 0.}, -3., diffuse_white);
    Plane<T> ceiling_plane(Vector<T, 3>{0., -1., 0.}, -3., diffuse_white);
    Sphere<T> light(Vector<T, 3>{0., 3., 3.}, 1., nullptr, emitter);

    // Build test scene
    Scene<T> scene;
    scene.push_back(&sphere_front);
    scene.push_back(&sphere_back);
    scene.push_back(&left_plane);
    scene.push_back(&right_plane);
    scene.push_back(&back_plane);
    scene.push_back(&front_plane);
    scene.push_back(&ground_plane);
    scene.push_back(&ceiling_plane);
    scene.push_back(&light);

    // Configure camera position and resolution
    std::size_t width = args.width;
    std::size_t height = args.height;
    Camera<T> cam(width, height);
    cam.look_at(Vector<T, 3>{0, 0, 0}, Vector<T, 3>{0, 0, 1});
    Vector<double, 3> *img = new Vector<double, 3>[width * height];

    // Configure path tracer sampling
    Pathtracer<T> tracer(args.absorb_prob, args.min_bounces);

    // Render test scene
    for (std::size_t y = 0; y < cam.height(); ++y) {
        for (std::size_t x = 0; x < cam.width(); ++x) {
            Vector<T, 3> pixel_radiance(0);
            for (std::size_t i = 0; i < args.samples; ++i) {
                Vector<T, 3> dir = cam.sample(x, y);
                Vector<T, 3, true> radiance = tracer.trace(scene, cam.eye(), dir);
                pixel_radiance += radiance.detach();
		// Uncomment to compute gradients
                // radiance.backward(Vec3(1));
            }
            img[y*width + x] = pixel_radiance / args.samples;
        }
        printf("% 5.2f%%\r", 100. * (y+1) / cam.height());
        fflush(stdout);
    }
    printf("\n");

    // Write radiance to file
    write_exr(args.output.c_str(), img, width, height);

    return 0;
}
