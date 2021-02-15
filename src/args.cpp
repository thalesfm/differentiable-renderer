#include <tclap/CmdLine.h>
#include "args.hpp"

namespace drt { namespace args {

bool parse(int argc, const char *const *argv, Args *args)
{
    TCLAP::CmdLine cmd("A simple differentiable path tracer", ' ', "0.1");
    TCLAP::ValueArg<int> samples_arg(
        "n", "samples",
        "Number of samples per pixel",
        false,
        4,
        "integer"
    );
    cmd.add(samples_arg);
    TCLAP::ValueArg<int> min_bounces_arg(
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
        args->samples = samples_arg.getValue();
        args->min_bounces = min_bounces_arg.getValue();
        args->absorb_prob = absorb_prob_arg.getValue();
        args->output = output_arg.getValue();
    } catch (const TCLAP::ArgException& e) {
        return false;
    }
    return true;
}

} }
