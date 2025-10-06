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

#define CHECK_CUDA(call)                                                                       
    do {                                                                                        
        cudaError_t status = (call);                                                            
        if (status != cudaSuccess) {                                                            
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(status)); 
        }                                                                                       
    } while (false)

#define CHECK_CUDNN(call)                                                                         
    do {                                                                                          
        cudnnStatus_t status = (call);                                                            
        if (status != CUDNN_STATUS_SUCCESS) {                                                     
            throw std::runtime_error(std::string("cuDNN error: ") + cudnnGetErrorString(status)); 
        }                                                                                         
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

int main(int argc, char **argv) {
    try {
        std::string processed_root = "data_processed";
        std::string features_root = "data_features";
        int device_id = 0;

        if (argc >= 2) {
            processed_root = argv[1];
        }
        if (argc >= 3) {
            features_root = argv[2];
        }
        if (argc >= 4) {
            device_id = std::stoi(argv[3]);
        }

        CHECK_CUDA(cudaSetDevice(device_id));

        auto samples = enumerate_processed(processed_root);
        if (samples.empty()) {
            std::cerr << "No processed samples found under " << processed_root << "\n";
            return 1;
        }

        fs::create_directories(features_root);

        const int batch = 1;
        const int in_channels = 3;
        const int in_height = 224;
        const int in_width = 224;
        const int conv1_out_channels = 16;
        const int conv2_out_channels = 32;
        const int kernel_size = 3;

        const int pool_window = 2;
        const int pool_stride = 2;

        const int pool1_height = in_height / 2;   // 112
        const int pool1_width = in_width / 2;     // 112
        const int pool2_height = pool1_height / 2;  // 56
        const int pool2_width = pool1_width / 2;    // 56

        const size_t input_elems = static_cast<size_t>(batch) * in_channels * in_height * in_width;
        const size_t conv1_elems = static_cast<size_t>(batch) * conv1_out_channels * in_height * in_width;
        const size_t pool1_elems = static_cast<size_t>(batch) * conv1_out_channels * pool1_height * pool1_width;
        const size_t conv2_elems = static_cast<size_t>(batch) * conv2_out_channels * pool1_height * pool1_width;
        const size_t pool2_elems = static_cast<size_t>(batch) * conv2_out_channels * pool2_height * pool2_width;

        float *d_input = nullptr;
        float *d_conv1_out = nullptr;
        float *d_pool1_out = nullptr;
        float *d_conv2_out = nullptr;
        float *d_pool2_out = nullptr;
        float *d_workspace = nullptr;

        CHECK_CUDA(cudaMalloc(&d_input, input_elems * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_conv1_out, conv1_elems * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_pool1_out, pool1_elems * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_conv2_out, conv2_elems * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_pool2_out, pool2_elems * sizeof(float)));

        // Weight and bias buffers
        std::vector<float> h_conv1_weights(conv1_out_channels * in_channels * kernel_size * kernel_size);
        std::vector<float> h_conv2_weights(conv2_out_channels * conv1_out_channels * kernel_size * kernel_size);
        std::vector<float> h_conv1_bias(conv1_out_channels, 0.0f);
        std::vector<float> h_conv2_bias(conv2_out_channels, 0.0f);
        initialise_conv_weights(h_conv1_weights, conv1_out_channels, in_channels, kernel_size);
        initialise_conv_weights(h_conv2_weights, conv2_out_channels, conv1_out_channels, kernel_size);

        float *d_conv1_weights = nullptr;
        float *d_conv2_weights = nullptr;
        float *d_conv1_bias = nullptr;
        float *d_conv2_bias = nullptr;

        CHECK_CUDA(cudaMalloc(&d_conv1_weights, h_conv1_weights.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_conv2_weights, h_conv2_weights.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_conv1_bias, h_conv1_bias.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_conv2_bias, h_conv2_bias.size() * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_conv1_weights, h_conv1_weights.data(),
                              h_conv1_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_conv2_weights, h_conv2_weights.data(),
                              h_conv2_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_conv1_bias, h_conv1_bias.data(),
                              h_conv1_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_conv2_bias, h_conv2_bias.data(),
                              h_conv2_bias.size() * sizeof(float), cudaMemcpyHostToDevice));

        cudnnHandle_t cudnn;
        CHECK_CUDNN(cudnnCreate(&cudnn));

        cudnnTensorDescriptor_t input_desc;
        cudnnTensorDescriptor_t conv1_out_desc;
        cudnnTensorDescriptor_t pool1_out_desc;
        cudnnTensorDescriptor_t conv2_out_desc;
        cudnnTensorDescriptor_t pool2_out_desc;
        cudnnFilterDescriptor_t conv1_filter_desc;
        cudnnFilterDescriptor_t conv2_filter_desc;
        cudnnConvolutionDescriptor_t conv1_desc;
        cudnnConvolutionDescriptor_t conv2_desc;
        cudnnTensorDescriptor_t conv1_bias_desc;
        cudnnTensorDescriptor_t conv2_bias_desc;
        cudnnActivationDescriptor_t relu_desc;
        cudnnPoolingDescriptor_t pool_desc;

        CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&conv1_out_desc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&pool1_out_desc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&conv2_out_desc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&pool2_out_desc));
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&conv1_filter_desc));
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&conv2_filter_desc));
        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv1_desc));
        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv2_desc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&conv1_bias_desc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&conv2_bias_desc));
        CHECK_CUDNN(cudnnCreateActivationDescriptor(&relu_desc));
        CHECK_CUDNN(cudnnCreatePoolingDescriptor(&pool_desc));

        CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               batch, in_channels, in_height, in_width));

        CHECK_CUDNN(cudnnSetFilter4dDescriptor(conv1_filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                               conv1_out_channels, in_channels, kernel_size, kernel_size));
        CHECK_CUDNN(cudnnSetFilter4dDescriptor(conv2_filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                               conv2_out_channels, conv1_out_channels, kernel_size, kernel_size));

        CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv1_desc,
                                                    /*pad_h*/ 1, /*pad_w*/ 1,
                                                    /*stride_h*/ 1, /*stride_w*/ 1,
                                                    /*dilation_h*/ 1, /*dilation_w*/ 1,
                                                    CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
        CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv2_desc,
                                                    1, 1,
                                                    1, 1,
                                                    1, 1,
                                                    CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

        CHECK_CUDNN(cudnnSetTensor4dDescriptor(conv1_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               batch, conv1_out_channels, in_height, in_width));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(pool1_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               batch, conv1_out_channels, pool1_height, pool1_width));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(conv2_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               batch, conv2_out_channels, pool1_height, pool1_width));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(pool2_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               batch, conv2_out_channels, pool2_height, pool2_width));

        CHECK_CUDNN(cudnnSetTensor4dDescriptor(conv1_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               1, conv1_out_channels, 1, 1));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(conv2_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               1, conv2_out_channels, 1, 1));

        CHECK_CUDNN(cudnnSetActivationDescriptor(relu_desc, CUDNN_ACTIVATION_RELU,
                                                 CUDNN_PROPAGATE_NAN, 0.0));

        CHECK_CUDNN(cudnnSetPooling2dDescriptor(pool_desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
                                                pool_window, pool_window,
                                                0, 0,
                                                pool_stride, pool_stride));

        cudnnConvolutionFwdAlgoPerf_t conv1_perf;
        cudnnConvolutionFwdAlgoPerf_t conv2_perf;
        int returned = 0;

        CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnn, input_desc, conv1_filter_desc, conv1_desc,
                                                           conv1_out_desc, 1, &returned, &conv1_perf));
        CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnn, pool1_out_desc, conv2_filter_desc, conv2_desc,
                                                           conv2_out_desc, 1, &returned, &conv2_perf));

        size_t workspace_bytes = 0;
        size_t conv1_ws = 0;
        size_t conv2_ws = 0;
        CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_desc, conv1_filter_desc, conv1_desc,
                                                             conv1_out_desc, conv1_perf.algo, &conv1_ws));
        CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, pool1_out_desc, conv2_filter_desc, conv2_desc,
                                                             conv2_out_desc, conv2_perf.algo, &conv2_ws));
        workspace_bytes = std::max(conv1_ws, conv2_ws);
        if (workspace_bytes > 0) {
            CHECK_CUDA(cudaMalloc(&d_workspace, workspace_bytes));
        }

        std::vector<std::string> manifest_lines;
        manifest_lines.reserve(samples.size() + 1);
        manifest_lines.emplace_back("class,raw_id,feature_path,mask_path,image_path");

        std::vector<float> h_input(input_elems);
        std::vector<float> h_output(pool2_elems);

        const float alpha = 1.0f;
        const float beta = 0.0f;

        for (size_t idx = 0; idx < samples.size(); ++idx) {
            const Sample &sample = samples[idx];
            h_input = load_image_chw(sample.image_path, in_height, in_width);

            CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), input_elems * sizeof(float), cudaMemcpyHostToDevice));

            CHECK_CUDNN(cudnnConvolutionForward(cudnn,
                                               &alpha,
                                               input_desc,
                                               d_input,
                                               conv1_filter_desc,
                                               d_conv1_weights,
                                               conv1_desc,
                                               conv1_perf.algo,
                                               d_workspace,
                                               workspace_bytes,
                                               &beta,
                                               conv1_out_desc,
                                               d_conv1_out));

            CHECK_CUDNN(cudnnAddTensor(cudnn,
                                       &alpha,
                                       conv1_bias_desc,
                                       d_conv1_bias,
                                       &alpha,
                                       conv1_out_desc,
                                       d_conv1_out));

            CHECK_CUDNN(cudnnActivationForward(cudnn,
                                               relu_desc,
                                               &alpha,
                                               conv1_out_desc,
                                               d_conv1_out,
                                               &beta,
                                               conv1_out_desc,
                                               d_conv1_out));

            CHECK_CUDNN(cudnnPoolingForward(cudnn,
                                             pool_desc,
                                             &alpha,
                                             conv1_out_desc,
                                             d_conv1_out,
                                             &beta,
                                             pool1_out_desc,
                                             d_pool1_out));

            CHECK_CUDNN(cudnnConvolutionForward(cudnn,
                                               &alpha,
                                               pool1_out_desc,
                                               d_pool1_out,
                                               conv2_filter_desc,
                                               d_conv2_weights,
                                               conv2_desc,
                                               conv2_perf.algo,
                                               d_workspace,
                                               workspace_bytes,
                                               &beta,
                                               conv2_out_desc,
                                               d_conv2_out));

            CHECK_CUDNN(cudnnAddTensor(cudnn,
                                       &alpha,
                                       conv2_bias_desc,
                                       d_conv2_bias,
                                       &alpha,
                                       conv2_out_desc,
                                       d_conv2_out));

            CHECK_CUDNN(cudnnActivationForward(cudnn,
                                               relu_desc,
                                               &alpha,
                                               conv2_out_desc,
                                               d_conv2_out,
                                               &beta,
                                               conv2_out_desc,
                                               d_conv2_out));

            CHECK_CUDNN(cudnnPoolingForward(cudnn,
                                             pool_desc,
                                             &alpha,
                                             conv2_out_desc,
                                             d_conv2_out,
                                             &beta,
                                             pool2_out_desc,
                                             d_pool2_out));

            CHECK_CUDA(cudaMemcpy(h_output.data(), d_pool2_out, pool2_elems * sizeof(float), cudaMemcpyDeviceToHost));

            fs::path class_dir = fs::path(features_root) / sample.class_name;
            fs::create_directories(class_dir);
            fs::path feature_path = class_dir / (sample.stem + "_features.npy");

            if (!write_npy(feature_path, h_output.data(), h_output.size(),
                            {static_cast<size_t>(conv2_out_channels), static_cast<size_t>(pool2_height), static_cast<size_t>(pool2_width)})) {
                std::cerr << "Failed to write feature map: " << feature_path << "\n";
            }

            std::ostringstream line;
            line << sample.class_name << ','
                 << sample.stem << ','
                 << relative_or_absolute(feature_path, features_root) << ','
                 << relative_or_absolute(sample.mask_path, processed_root) << ','
                 << relative_or_absolute(sample.image_path, processed_root);
            manifest_lines.push_back(line.str());

            if ((idx + 1) % 50 == 0 || idx + 1 == samples.size()) {
                std::cout << "Processed " << (idx + 1) << "/" << samples.size() << " feature maps.\n";
            }
        }

        fs::path manifest_path = fs::path(features_root) / "manifest.csv";
        std::ofstream manifest(manifest_path);
        for (const auto &line : manifest_lines) {
            manifest << line << '\n';
        }
        std::cout << "Feature manifest written to " << manifest_path << "\n";

        // Cleanup
        if (d_workspace) {
            cudaFree(d_workspace);
        }
        cudaFree(d_pool2_out);
        cudaFree(d_conv2_out);
        cudaFree(d_pool1_out);
        cudaFree(d_conv1_out);
        cudaFree(d_input);
        cudaFree(d_conv1_weights);
        cudaFree(d_conv2_weights);
        cudaFree(d_conv1_bias);
        cudaFree(d_conv2_bias);

        cudnnDestroyPoolingDescriptor(pool_desc);
        cudnnDestroyActivationDescriptor(relu_desc);
        cudnnDestroyTensorDescriptor(conv2_bias_desc);
        cudnnDestroyTensorDescriptor(conv1_bias_desc);
        cudnnDestroyConvolutionDescriptor(conv2_desc);
        cudnnDestroyConvolutionDescriptor(conv1_desc);
        cudnnDestroyFilterDescriptor(conv2_filter_desc);
        cudnnDestroyFilterDescriptor(conv1_filter_desc);
        cudnnDestroyTensorDescriptor(pool2_out_desc);
        cudnnDestroyTensorDescriptor(conv2_out_desc);
        cudnnDestroyTensorDescriptor(pool1_out_desc);
        cudnnDestroyTensorDescriptor(conv1_out_desc);
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroy(cudnn);

        CHECK_CUDA(cudaDeviceSynchronize());
        std::cout << "Feature extraction completed successfully." << std::endl;
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "Fatal error: " << ex.what() << std::endl;
        return 1;
    }
}
