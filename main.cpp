#include <ImfRgbaFile.h>
#include <ImfArray.h>
#include <ImfInputFile.h>
#include <iostream>
#include <eigen3/Eigen/Core>
#include <chrono>

constexpr size_t output_width = 256;
constexpr size_t output_height = 128;
constexpr float N1 = 100;
constexpr float N2 = 100;
constexpr float thetaStep = 1.0 / N1;
constexpr float phiStep = 1.0 / N2;

using Eigen::Vector2f;
using Eigen::Vector3f;


Vector3f sphericalToCartesian(float theta, float phi)
{
    auto sin_theta = sin(theta);
    auto cos_theta = cos(theta);
    auto sin_phi = sin(phi);
    auto cos_phi = cos(phi);
    return Vector3f(
        sin_theta * cos_phi,
        sin_theta * sin_phi,
        cos_theta);
}

Vector2f cartesianToSpherical(const Vector3f& v)
{
    Vector2f result(
        std::acos(v.z()),
        std::atan2(v.y(), v.x()));
    if (result.y() < 0)
        result.y() += 2 * M_PI;
    return result;
}

Vector2f uvToSpherical(float u, float v)
{
    auto phi = u * 2.0 * M_PI;
    auto theta = v * M_PI;
    return Vector2f(theta, phi);
}

Vector2f sphericalToUV(float theta, float phi)
{
    auto u = phi * M_2_PI;
    auto v = theta * M_1_PI;
    return Vector2f(u, v);
}

Vector2f cartesianToUV(const Vector3f& v)
{
    auto sph = cartesianToSpherical(v);
    return sphericalToUV(sph[0], sph[1]);
}

Vector3f uvToCartesian(float u, float v)
{
    auto sph = uvToSpherical(u, v);
    return sphericalToCartesian(sph[0], sph[1]);
}


void read(std::string fileName, Imf::Array2D<Imf::Rgba> &pixels, size_t &width, size_t &height)
{
    Imf::RgbaInputFile file(fileName.c_str());
    Imath::Box2i dw = file.dataWindow();

    width = dw.max.x - dw.min.x + 1;
    height = dw.max.y - dw.min.y + 1;
    pixels.resizeErase(height, width);

    file.setFrameBuffer(&pixels[0][0] - dw.min.x - dw.min.y * width, 1, width);
    file.readPixels(dw.min.y, dw.max.y);
}

void process(Imf::Array2D<Imf::Rgba> &input_pixels, Imf::Array2D<Imf::Rgba>& output_pixels)
{
    std::array<std::array<float, output_width>, output_height> R{};
    std::array<std::array<float, output_width>, output_height> G{};
    std::array<std::array<float, output_width>, output_height> B{};

    std::array<std::array<Vector3f, output_width>, output_height> output_dir_table{};
    for (size_t output_row = 0; output_row < output_pixels.height(); output_row++)
    {
        for (size_t output_col = 0; output_col < output_pixels.width(); output_col++)
        {
            auto output_u = (output_col + 0.5) / output_pixels.width();
            auto output_v = (output_row + 0.5) / output_pixels.height();
            output_dir_table[output_row][output_col] = uvToCartesian(output_u, output_v);
        }
    }

    auto pdfSample = 2*M_PI / (N1 * N2);
#pragma omp parallel for

    for (size_t thetaIdx = 0; thetaIdx < N1; thetaIdx++)
    {
        for (size_t phiIdx = 0; phiIdx < N2; phiIdx++)
        {
            auto theta = thetaIdx * thetaStep * M_PI;
            auto phi = phiIdx * phiStep * 2*M_PI;
            auto u = phiIdx / N2;
            auto v = thetaIdx / N1;
            auto dir = uvToCartesian(u, v);
            // auto uv = sphericalToUV(theta, phi);
            // auto dir = sphericalToCartesian(theta, phi);
            // auto u = uv[0];
            // auto v = uv[1];
            float row_f = v * (input_pixels.height()-1);
            float col_f = u * (input_pixels.width()-1);
            size_t row = row_f;
            size_t col = col_f;

            // lerp or closet neighbor???
            auto color0 = input_pixels[row][col];
            // auto color1 = input_pixels[row][col+1];
            // auto color2 = input_pixels[row+1][col];
            // auto color3 = input_pixels[row+1][col+1];
            auto sin_theta = sin(theta);
            auto cos_theta = cos(theta);
            double factor = sin_theta  * pdfSample;
            auto r = color0.r;
            auto g = color0.g;
            auto b = color0.b;

            // auto r = 0.25 * (color0.r + color1.r + color2.r + color3.r);
            // auto g = 0.25 * (color0.g + color1.g + color2.g + color3.g);
            // auto b = 0.25 * (color0.b + color1.b + color2.b + color3.b);
            // double factor = pdfSample;
            // double factor = sin_theta ;
            // auto factor = sin_theta * cos_theta * pdfSample;
            // auto intensity = Imf::Rgba(color.r * sin_theta, color.g * sin_theta,color.b * sin_theta,1);
            // auto intensity = Imf::Rgba(color.r * factor, color.g * factor,color.b * factor,1);

            for (size_t output_row = 0; output_row < output_pixels.height(); output_row++)
            {
                for (size_t output_col = 0; output_col < output_pixels.width(); output_col++)
                {
                    auto _cos = output_dir_table[output_row][output_col].dot(dir);
                    if (_cos > 0)
                    {
                        R[output_row][output_col] += std::max(0.0, r * factor*_cos);
                        G[output_row][output_col] += std::max(0.0, g * factor*_cos);
                        B[output_row][output_col] += std::max(0.0, b * factor*_cos);
                    }
                }
            }
        }
    }
    
    for (size_t output_row = 0; output_row < output_pixels.height(); output_row++)
    {
        for (size_t output_col = 0; output_col < output_pixels.width(); output_col++)
        {
            output_pixels[output_row][output_col] = Imf::Rgba(R[output_row][output_col],G[output_row][output_col],B[output_row][output_col],1);
        }
    }
}

void write(const Imf::Array2D<Imf::Rgba> &pixels, std::string fileName)
{

    try
    {
        Imf::RgbaOutputFile file(fileName.c_str(), pixels.width(), pixels.height(), Imf::WRITE_RGB);
        file.setFrameBuffer(&pixels[0][0], 1, pixels.width());
        file.writePixels(pixels.height());
    }
    catch (const std::exception &e)
    {
        std::cerr << "error writing image file :" << e.what() << std::endl;
    }
}

int main()
{
    Imf::Array2D<Imf::Rgba> pixels;
    size_t input_width;
    size_t input_height;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();


    // read("/Users/lkj/Git/CG/HDR/uffizi-large.exr", pixels, input_width, input_height);
    read("/Users/lkj/Git/CG/HDR/pisa.exr", pixels, input_width, input_height);

    Imf::Array2D<Imf::Rgba> output_pixels(output_height,output_width);
    process(pixels, output_pixels);
    // write(output_pixels, "meow.exr");
    write(output_pixels, "meow2.exr");

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;
    return 0;
}