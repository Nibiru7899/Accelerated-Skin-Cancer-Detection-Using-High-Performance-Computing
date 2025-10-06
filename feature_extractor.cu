#include <cuda_runtime.h>
#include <cudnn.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#define CHECK_CUDA(call)                                                                       \
    do {                                                                                        \
        cudaError_t status = (call);                                                            \
        if (status != cudaSuccess) {                                                            \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(status)); \
        }                                                                                       \
    } while (false)

#define CHECK_CUDNN(call)                                                                         \
    do {                                                                                          \
        cudnnStatus_t status = (call);                                                            \
        if (status != CUDNN_STATUS_SUCCESS) {                                                     \
            throw std::runtime_error(std::string("cuDNN error: ") + cudnnGetErrorString(status)); \
        }                                                                                         \
    } while (false)

struct Sample {
    fs::path image_path;
    fs::path mask_path;
    std::string class_name;
    std::string stem;  // basename without _image/_mask suffix.
};

namespace {

std::vector<Sample> enumerate_processed(const fs::path &processed_root) {
    if (!fs::exists(processed_root)) {
        throw std::runtime_error("Processed dataset root does not exist: " + processed_root.string());
    }

    std::vector<Sample> samples;

    for (const auto &class_entry : fs::directory_iterator(processed_root)) {
        if (!class_entry.is_directory()) {
            continue;
        }
        const std::string class_name = class_entry.path().filename().string();
        for (const auto &file_entry : fs::directory_iterator(class_entry.path())) {
            if (!file_entry.is_regular_file()) {
                continue;
            }
            const fs::path &image_path = file_entry.path();
            const std::string filename = image_path.filename().string();
            if (filename.size() < 10 || filename.rfind("_image.png") == std::string::npos) {
                continue;
            }
            const std::string stem = filename.substr(0, filename.size() - std::string("_image.png").size());
            const fs::path mask_path = class_entry.path() / (stem + "_mask.png");
            if (!fs::exists(mask_path)) {
                std::cerr << "Warning: mask missing for " << image_path << ", skipping.\n";
                continue;
            }
            samples.push_back({image_path, mask_path, class_name, stem});
        }
    }

    std::sort(samples.begin(), samples.end(), [](const Sample &a, const Sample &b) {
        if (a.class_name == b.class_name) {
            return a.stem < b.stem;
        }
        return a.class_name < b.class_name;
    });

    return samples;
}

std::vector<float> load_image_chw(const fs::path &path, int height, int width) {
    cv::Mat bgr = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (bgr.empty()) {
        throw std::runtime_error("Failed to read image: " + path.string());
    }
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(width, height), 0, 0, cv::INTER_AREA);
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0f / 255.0f);

    std::vector<float> tensor(3 * height * width);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < height; ++y) {
            const float *row = rgb.ptr<float>(y);
            for (int x = 0; x < width; ++x) {
                tensor[c * height * width + y * width + x] = row[x * 3 + c];
            }
        }
    }
    return tensor;
}

void initialise_conv_weights(std::vector<float> &weights, int out_channels, int in_channels, int kernel) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(in_channels * kernel * kernel));
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int ky = 0; ky < kernel; ++ky) {
                for (int kx = 0; kx < kernel; ++kx) {
                    const size_t idx = (((oc * in_channels) + ic) * kernel + ky) * kernel + kx;
                    const float angle = static_cast<float>((oc + 1) * (ic + 1) * (ky + 1) * (kx + 1));
                    weights[idx] = scale * std::sin(angle);
                }
            }
        }
    }
}

bool write_npy(const fs::path &path, const float *data, size_t count, const std::vector<size_t> &shape) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        return false;
    }
    std::ostringstream oss;
    oss << "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        oss << shape[i];
        if (i + 1 < shape.size()) {
            oss << ", ";
        }
    }
    if (shape.size() == 1) {
        oss << ",";
    }
    oss << "), }";
    std::string header = oss.str();
    header.push_back(' ');
    while ((10 + header.size()) % 16 != 0) {
        header.push_back(' ');
    }
    header.back() = '\n';

    const uint16_t header_size = static_cast<uint16_t>(header.size());
    ofs.write("\x93NUMPY", 6);
    char version[2] = {1, 0};
    ofs.write(version, 2);
    ofs.write(reinterpret_cast<const char *>(&header_size), 2);
    ofs.write(header.data(), static_cast<std::streamsize>(header.size()));
    ofs.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(count * sizeof(float)));
    return ofs.good();
}

std::string relative_or_absolute(const fs::path &target, const fs::path &base) {
    try {
        return fs::relative(target, base).string();
    } catch (...) {
        return target.string();
    }
}

}  // namespace

// NOTE: This is Part 1 of the feature extractor
// Part 2 should contain the main() function with CUDA/cuDNN implementation
// Merge both parts to create the complete feature_extractor_cudnn.cu file