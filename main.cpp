#include <ImfRgbaFile.h>
#include <ImfArray.h>
#include <ImfInputFile.h>
#include <iostream>
#include <chrono>
#include <array>

constexpr size_t output_width = 256;
constexpr size_t output_height = 128;
constexpr float N1 = 100;
constexpr float N2 = 100;
constexpr float thetaStep = 1.0 / N1;
constexpr float phiStep = 1.0 / N2;
 

std::array<float,3> sphericalToCartesian(float theta, float phi)
{
    auto sin_theta = sin(theta);
    auto cos_theta = cos(theta);
    auto sin_phi = sin(phi);
    auto cos_phi = cos(phi);
    return {
        sin_theta * cos_phi,
        sin_theta * sin_phi,
        cos_theta
    };
}

std::array<float,2> cartesianToSpherical(const std::array<float,3>& v)
{
    std::array<float,2> result{
        std::acos(v[2]),
        std::atan2(v[2], v[0])
    };
    if (result[2] < 0)
        result[2] += 2 * M_PI;
    return result;
}

std::array<float,2> uvToSpherical(float u, float v)
{
    float phi = u * 2.0 * M_PI;
    float theta = v * M_PI;
    return std::array<float,2>{theta, phi};
}

std::array<float,2> sphericalToUV(float theta, float phi)
{
    float u = phi * M_2_PI;
    float v = theta * M_1_PI;
    return std::array<float,2>{u, v};
}

std::array<float,2> cartesianToUV(const std::array<float,3>& v)
{
    auto sph = cartesianToSpherical(v);
    return sphericalToUV(sph[0], sph[1]);
}

std::array<float,3> uvToCartesian(float u, float v)
{
    auto sph = uvToSpherical(u, v);
    return sphericalToCartesian(sph[0], sph[1]);
}

float dotProduct(std::array<float,3> a, std::array<float,3> b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
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

    std::array<std::array<std::array<float,3>, output_width>, output_height> output_dir_table{};
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
     
            float row_f = v * (input_pixels.height()-1);
            float col_f = u * (input_pixels.width()-1);
            size_t row = row_f;
            size_t col = col_f;

            // TODO: Lerp or nearest neighbor ???
            auto intensity = input_pixels[row][col];
            auto sin_theta = sin(theta);
            auto factor = sin_theta * pdfSample;
            auto r = intensity.r;
            auto g = intensity.g;
            auto b = intensity.b;

            for (size_t output_row = 0; output_row < output_pixels.height(); output_row++)
            {
                for (size_t output_col = 0; output_col < output_pixels.width(); output_col++)
                {
                    auto cos_diff = dotProduct(output_dir_table[output_row][output_col], dir);
                    if (cos_diff > 0)
                    {
                        R[output_row][output_col] += std::max(0.0, r * factor*cos_diff);
                        G[output_row][output_col] += std::max(0.0, g * factor*cos_diff);
                        B[output_row][output_col] += std::max(0.0, b * factor*cos_diff);
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


    read("/Users/lkj/Git/CG/HDR/uffizi-large.exr", pixels, input_width, input_height);
    // read("/Users/lkj/Git/CG/HDR/pisa.exr", pixels, input_width, input_height);

    Imf::Array2D<Imf::Rgba> output_pixels(output_height,output_width);
    process(pixels, output_pixels);
    write(output_pixels, "diffuse_uffizi.exr");
    // write(output_pixels, "diffuse_pisa.exr");

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;
    return 0;
}