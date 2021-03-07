#pragma once

#include <ImfRgba.h>
#include <ImfRgbaFile.h>
#include "drt/vector.hpp"

namespace drt {

template <typename T>
inline void write_exr(const char *fname,
                      const Vector<T, 3> *data,
                      std::size_t width,
                      std::size_t height)
{
    std::vector<Imf::Rgba> pixels;
    pixels.reserve(width * height);
    for (std::size_t i = 0; i < height; ++i) {
        for (std::size_t j = 0; j < width; ++j) {
            Vector<T, 3> rgb = data[i*width + j];
            pixels.emplace_back(double(rgb[0]), double(rgb[1]), double(rgb[2]));
        }
    }
    Imf::RgbaOutputFile file(fname, width, height, Imf::WRITE_RGBA);
    file.setFrameBuffer(pixels.data(), 1, width);
    file.writePixels(height);
}

} // namespace drt
