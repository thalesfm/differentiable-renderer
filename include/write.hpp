#pragma once

#include <ImfRgba.h>
#include <ImfRgbaFile.h>
#include "vector.hpp"

namespace drt {

template <typename T>
inline void write_exr(const char *fname, const Vector<T, 3> *data,
    std::size_t width, std::size_t height)
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

/*
void write_json(const char *fname, cube& img)
{
    FILE *fp = fopen(fname, "w");
    fprintf(fp, "[\n");
    for (int i = 0; i < img.n_rows; ++i) {
        fprintf(fp, "  [\n");
        for (int j = 0; j < img.n_cols; ++j) {
            fprintf(fp, "    [");
            for (int k = 0; k < img.n_slices; ++k) {
                fprintf(fp, "%f", img(i, j, k));
                if (k != img.n_slices - 1) {
                    fprintf(fp, ", ");
                }
            }
            fprintf(fp, "]");
            if (j != img.n_cols - 1) {
                fprintf(fp, ", ");
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "  ]");
        if (i != img.n_rows - 1) {
            fprintf(fp, ", ");
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "]\n");
    fclose(fp);
}
*/

}
