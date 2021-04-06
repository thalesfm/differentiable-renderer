#pragma once

#include <cstddef>
#include <tclap/CmdLine.h>

namespace drt {

struct Args {
    std::size_t width;
    std::size_t height;
    std::size_t samples;
    std::size_t min_bounces;
    double absorb_prob;
    std::string output;
};

inline bool parse_args(int argc, const char *const *argv, Args *args)
{
    TCLAP::CmdLine cmd("A simple differentiable path tracer", ' ', "0.1");
    TCLAP::ValueArg<std::size_t> width_arg(
        "x", "width",
        "Output image width",
        false,
        640,
        "integer"
    );
    cmd.add(width_arg);
    TCLAP::ValueArg<std::size_t> height_arg(
        "y", "height",
        "Output image height",
        false,
        480,
        "integer"
    );
    cmd.add(height_arg);
    TCLAP::ValueArg<std::size_t> samples_arg(
        "n", "samples",
        "Number of samples per pixel",
        false,
        100,
        "integer"
    );
    cmd.add(samples_arg);
    TCLAP::ValueArg<std::size_t> min_bounces_arg(
        "b", "min-bounces",
        "Min. number of light bounces",
        false,
        1,
        "integer"
    );
    cmd.add(min_bounces_arg);
    TCLAP::ValueArg<double> absorb_prob_arg(
        "p", "absorb-prob",
        "Ray absorbption prob. per bounce (after min. bounces)",
        false,
        0.5,
        "number"
    );
    cmd.add(absorb_prob_arg);
    TCLAP::ValueArg<std::string> output_arg(
        "o", "output",
        "Output path",
        true,
        "",
        "string"
    );
    cmd.add(output_arg);
    try {
        cmd.parse(argc, argv);
        args->width = width_arg.getValue();
        args->height = height_arg.getValue();
        args->samples = samples_arg.getValue();
        args->min_bounces = min_bounces_arg.getValue();
        args->absorb_prob = absorb_prob_arg.getValue();
        args->output = output_arg.getValue();
    } catch (const TCLAP::ArgException& e) {
        return false;
    }
    return true;
}

}
