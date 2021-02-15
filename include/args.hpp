#pragma once

namespace drt { namespace args {

struct Args {
    int samples;
    int min_bounces;
    double absorb_prob;
    std::string output;
};

bool parse(int argc, const char *const *argv, Args *args);

} }
