// High-performance preprocessing of dermoscopic images using OpenMP + OpenCV.
// The tool ingests raw ISIC-style folders containing *_Orig.(jpg|png) images and
// their *_GT.(png|jpg) binary masks exported straight from ``data_raw/`` and
// emits resized, hair-removed, colour-normalised PNGs under ``data_processed/``.

// Include OpenCV library for image processing operations
#include <opencv2/opencv.hpp>
// Include filesystem library for directory and file path operations
#include <filesystem>
// Include vector for dynamic arrays
#include <vector>
// Include string for string manipulation
#include <string>
// Include iostream for input/output operations
#include <iostream>
// Include algorithm for standard algorithms like find, transform
#include <algorithm>
// Include optional for optional return values
#include <optional>
// Include iomanip for formatted output
#include <iomanip>

// Include OpenMP library for parallel processing
#include <omp.h>

// Create an alias 'fs' for std::filesystem namespace for convenience
namespace fs = std::filesystem;

// Anonymous namespace to encapsulate helper functions and structures
namespace {

// Structure to hold a pair of image and mask paths with metadata
struct SamplePair {
    fs::path image_path;      // Path to the original dermoscopic image
    fs::path mask_path;       // Path to the corresponding ground truth mask
    std::string class_name;   // Class name (e.g., "Benign" or "Malignant")
    std::string raw_id;       // Raw identifier extracted from filename
};

// Small epsilon value to prevent division by zero in normalization
constexpr double kEps = 1e-6;

// Function to check if a file path has one of the specified extensions
bool has_extension(const fs::path &path, const std::vector<std::string> &exts) {
    // Extract the file extension from the path and convert to string
    std::string ext = path.extension().string();
    // Transform the extension to lowercase for case-insensitive comparison
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        // Convert each character to lowercase
        return static_cast<char>(std::tolower(c));
    });
    // Check if the lowercase extension exists in the provided list of extensions
    return std::find(exts.begin(), exts.end(), ext) != exts.end();
}

// Function to find a matching ground truth mask for a given image stem
std::optional<fs::path> find_matching_mask(const fs::path &class_dir, const std::string &stem) {
    // Static vector of supported mask file extensions
    static const std::vector<std::string> mask_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"};
    // Iterate through each supported mask extension
    for (const auto &ext : mask_exts) {
        // Construct the candidate mask path by combining directory, stem, "_GT" suffix, and extension
        fs::path candidate = class_dir / (stem + "_GT" + ext);
        // Check if the candidate mask file exists in the filesystem
        if (fs::exists(candidate)) {
            // Return the found mask path wrapped in an optional
            return candidate;
        }
    }
    // Return empty optional if no matching mask was found
    return std::nullopt;
}

// Function to collect all valid image-mask pairs from the raw dataset directory
std::vector<SamplePair> collect_pairs(const fs::path &root_dir) {
    // Static vector of supported image file extensions
    static const std::vector<std::string> image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"};
    // Vector to store all discovered image-mask pairs
    std::vector<SamplePair> pairs;

    // Check if the root directory exists
    if (!fs::exists(root_dir)) {
        // Throw an error if the root directory doesn't exist
        throw std::runtime_error("Raw dataset root does not exist: " + root_dir.string());
    }

    // Iterate through each entry (class directory) in the root directory
    for (const auto &class_entry : fs::directory_iterator(root_dir)) {
        // Skip if the entry is not a directory
        if (!class_entry.is_directory()) {
            continue;
        }
        // Get the path of the class directory (e.g., "Benign" or "Malignant")
        fs::path class_dir = class_entry.path();
        // Extract the class name from the directory name
        std::string class_name = class_dir.filename().string();

        // Iterate through each file in the class directory
        for (const auto &file_entry : fs::directory_iterator(class_dir)) {
            // Skip if the entry is not a regular file
            if (!file_entry.is_regular_file()) {
                continue;
            }
            // Get the full path of the file
            fs::path file_path = file_entry.path();
            // Check if the file has a valid image extension
            if (!has_extension(file_path, image_exts)) {
                continue;
            }
            // Extract the filename as a string
            const std::string filename = file_path.filename().string();
            // Check if the filename contains "_Orig" to identify original images
            if (filename.find("_Orig") == std::string::npos) {
                continue;
            }

            // Get the file stem (filename without extension)
            std::string stem = file_path.stem().string();
            // Find the position of "_Orig" in the stem
            size_t pos = stem.find("_Orig");
            // Skip if "_Orig" is not found (should not happen due to earlier check)
            if (pos == std::string::npos) {
                continue;
            }
            // Extract the raw ID by taking the substring before "_Orig"
            std::string raw_id = stem.substr(0, pos);
            // Try to find a matching ground truth mask for this image
            auto mask_path_opt = find_matching_mask(class_dir, raw_id);
            // If no mask was found, print a warning and skip this image
            if (!mask_path_opt) {
                std::cerr << "Warning: mask missing for " << file_path << ", skipping.\n";
                continue;
            }

            // Add the valid image-mask pair to the collection
            pairs.push_back({file_path, *mask_path_opt, class_name, raw_id});
        }
    }

    // Sort the pairs for deterministic processing order
    std::sort(pairs.begin(), pairs.end(), [](const SamplePair &a, const SamplePair &b) {
        // If both samples belong to the same class
        if (a.class_name == b.class_name) {
            // Sort by raw_id within the same class
            return a.raw_id < b.raw_id;
        }
        // Otherwise, sort by class name
        return a.class_name < b.class_name;
    });

    // Return the collected and sorted pairs
    return pairs;
}

// Function to remove hair artifacts from dermoscopic images using morphological operations
cv::Mat remove_hair(const cv::Mat &bgr, int kernel = 17, double thresh_val = 10.0) {
    // Create a matrix to hold the grayscale version of the image
    cv::Mat gray;
    // Convert the BGR image to grayscale for morphological processing
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    // Create a rectangular structuring element for morphological operations
    cv::Mat kernel_mat = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel, kernel));
    // Create a matrix to hold the result of the blackhat operation
    cv::Mat blackhat;
    // Apply blackhat morphological operation to detect dark hair strands on lighter skin
    cv::morphologyEx(gray, blackhat, cv::MORPH_BLACKHAT, kernel_mat);
    // Create a matrix to hold the thresholded binary mask
    cv::Mat thresh;
    // Apply binary threshold to create a mask of detected hair regions
    cv::threshold(blackhat, thresh, thresh_val, 255, cv::THRESH_BINARY);
    // Create a matrix to hold the inpainted result
    cv::Mat inpainted;
    // Inpaint the original image using the hair mask to remove hair artifacts
    cv::inpaint(bgr, thresh, inpainted, 1, cv::INPAINT_TELEA);
    // Return the hair-removed image
    return inpainted;
}

// Function to normalize the color distribution of an RGB image for consistent appearance
cv::Mat normalise_colour(const cv::Mat &rgb) {
    // Create a matrix to hold the floating-point version of the image
    cv::Mat float_img;
    // Convert the RGB image from 8-bit unsigned integer to 32-bit float, scaling to [0, 1]
    rgb.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    // Create Scalar objects to hold the mean and standard deviation of the image
    cv::Scalar mean, stddev;
    // Calculate the mean and standard deviation for each channel
    cv::meanStdDev(float_img, mean, stddev);

    // Create a vector to hold the three color channels separately
    std::vector<cv::Mat> channels(3);
    // Split the float image into separate R, G, B channels
    cv::split(float_img, channels);
    // Iterate through each of the three color channels
    for (int c = 0; c < 3; ++c) {
        // Standardize each channel by subtracting mean and dividing by standard deviation (z-score normalization)
        channels[c] = (channels[c] - mean[c]) / (stddev[c] + kEps);
    }
    // Create a matrix to hold the merged standardized channels
    cv::Mat merged;
    // Merge the three standardized channels back into a single image
    cv::merge(channels, merged);
    // Normalize the image to the range [0, 1] using min-max normalization
    cv::normalize(merged, merged, 0.0, 1.0, cv::NORM_MINMAX);

    // Create a matrix to hold the final 8-bit unsigned integer output
    cv::Mat out_uint8;
    // Convert the normalized float image back to 8-bit format, scaling to [0, 255]
    merged.convertTo(out_uint8, CV_8UC3, 255.0);
    // Return the color-normalized image
    return out_uint8;
}

// Function to write an OpenCV matrix as a PNG image file
bool write_png(const fs::path &path, const cv::Mat &mat, int thread_id) {
    // Attempt to write the matrix to the specified path as an image file
    if (!cv::imwrite(path.string(), mat)) {
        // If writing fails, output an error message with the thread ID and path
        std::cerr << "[Thread " << thread_id << "] Failed to write: " << path << "\n";
        // Return false to indicate failure
        return false;
    }
    // Return true to indicate successful writing
    return true;
}

// End of anonymous namespace
} // namespace

// Main function - entry point of the program
int main(int argc, char **argv) {
    // Initialize default input directory for raw image data
    std::string input_root = "data_raw";
    // Initialize default output directory for processed images
    std::string output_root = "data_processed";
    // Initialize default target width for resized images (224x224 for many neural networks)
    int width = 224;
    // Initialize default target height for resized images
    int height = 224;
    // Get the maximum number of threads available on the system
    int num_threads = omp_get_max_threads();

    // Check if at least 3 command-line arguments are provided (program name + 2 paths)
    if (argc >= 3) {
        // Set input root directory from first command-line argument
        input_root = argv[1];
        // Set output root directory from second command-line argument
        output_root = argv[2];
    }
    // Check if at least 5 command-line arguments are provided (includes dimensions)
    if (argc >= 5) {
        // Set target width from third command-line argument
        width = std::stoi(argv[3]);
        // Set target height from fourth command-line argument
        height = std::stoi(argv[4]);
    }
    // Check if at least 6 command-line arguments are provided (includes thread count)
    if (argc >= 6) {
        // Set number of threads from fifth argument, ensuring at least 1 thread
        num_threads = std::max(1, std::stoi(argv[5]));
    }

    // Set the number of OpenMP threads to be used for parallel processing
    omp_set_num_threads(num_threads);

    // Collect all valid image-mask pairs from the input directory
    std::vector<SamplePair> pairs = collect_pairs(input_root);
    // Check if any valid pairs were found
    if (pairs.empty()) {
        // Print error message if no pairs were discovered
        std::cerr << "No valid image/mask pairs discovered under " << input_root << "." << std::endl;
        // Exit with error code 1
        return 1;
    }

    // Print the number of discovered pairs to inform the user
    std::cout << "Discovered " << pairs.size() << " image/mask pairs under '" << input_root << "'.\n";
    // Print the number of threads that will be used for parallel processing
    std::cout << "Using " << num_threads << " threads." << std::endl;

    // Create the output root directory and any necessary parent directories
    fs::create_directories(output_root);

    // Record the start time for performance measurement
    double start_time = omp_get_wtime();

    // Parallel for loop: process all image-mask pairs in parallel with dynamic scheduling
    #pragma omp parallel for schedule(dynamic)
    for (std::size_t idx = 0; idx < pairs.size(); ++idx) {
        // Get a constant reference to the current sample pair
        const SamplePair &sample = pairs[idx];
        // Get the ID of the current OpenMP thread for logging purposes
        int thread_id = omp_get_thread_num();

        // Read the original dermoscopic image in BGR color format
        cv::Mat bgr = cv::imread(sample.image_path.string(), cv::IMREAD_COLOR);
        // Check if the image was successfully loaded
        if (bgr.empty()) {
            // Critical section to ensure thread-safe console output
            #pragma omp critical(log)
            // Print error message if image reading failed
            std::cerr << "[Thread " << thread_id << "] Failed to read image: " << sample.image_path << "\n";
            // Skip to the next iteration
            continue;
        }

        // Read the ground truth mask in grayscale format
        cv::Mat mask = cv::imread(sample.mask_path.string(), cv::IMREAD_GRAYSCALE);
        // Check if the mask was successfully loaded
        if (mask.empty()) {
            // Critical section to ensure thread-safe console output
            #pragma omp critical(log)
            // Print error message if mask reading failed
            std::cerr << "[Thread " << thread_id << "] Failed to read mask: " << sample.mask_path << "\n";
            // Skip to the next iteration
            continue;
        }

        // Apply hair removal algorithm to the image
        cv::Mat hairless = remove_hair(bgr);
        // Create a matrix to hold the resized RGB image
        cv::Mat resized_rgb;
        // Resize the hair-removed image to the target dimensions using area interpolation
        cv::resize(hairless, resized_rgb, cv::Size(width, height), 0, 0, cv::INTER_AREA);
        // Convert the color space from BGR (OpenCV default) to RGB
        cv::cvtColor(resized_rgb, resized_rgb, cv::COLOR_BGR2RGB);

        // Apply color normalization to the resized RGB image
        cv::Mat processed_rgb = normalise_colour(resized_rgb);

        // Create a matrix to hold the resized mask
        cv::Mat resized_mask;
        // Resize the mask to the target dimensions using nearest-neighbor interpolation
        cv::resize(mask, resized_mask, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
        // Create a matrix to hold the binary mask
        cv::Mat binary_mask;
        // Apply binary thresholding to ensure the mask contains only 0 or 255 values
        cv::threshold(resized_mask, binary_mask, 127, 255, cv::THRESH_BINARY);

        // Construct the output class directory path (e.g., data_processed/Benign)
        fs::path class_dir = fs::path(output_root) / sample.class_name;
        // Create a lowercase version of the class name for consistent filename format
        std::string class_lower = sample.class_name;
        // Transform the class name to lowercase character by character
        std::transform(class_lower.begin(), class_lower.end(), class_lower.begin(), [](unsigned char c) {
            // Convert each character to lowercase
            return static_cast<char>(std::tolower(c));
        });
        // Construct the file stem path (e.g., data_processed/Benign/benign_0)
        fs::path stem = class_dir / (class_lower + "_" + sample.raw_id);

        // Create a scope block for the critical section
        {
            // Critical section to ensure thread-safe directory creation
            #pragma omp critical(dir)
            // Create the class directory if it doesn't already exist
            fs::create_directories(class_dir);
        }

    // Construct the output path for the processed image
    fs::path image_out = stem;
    // Append the "_image.png" suffix to create the final image filename
    image_out += "_image.png";
        // Construct the output path for the processed mask
        fs::path mask_out = stem;
        // Append the "_mask.png" suffix to create the final mask filename
        mask_out += "_mask.png";

    // Create a matrix to hold the BGR version of the processed image for saving
    cv::Mat bgr_output;
    // Convert the processed RGB image back to BGR format (OpenCV's default for saving)
    cv::cvtColor(processed_rgb, bgr_output, cv::COLOR_RGB2BGR);
    // Write the processed image to disk as a PNG file
    write_png(image_out, bgr_output, thread_id);
        // Write the binary mask to disk as a PNG file
        write_png(mask_out, binary_mask, thread_id);

        // Check if this is thread 0 and if it's time to report progress (every 50 samples or at the end)
        if (thread_id == 0 && (idx % 50 == 0 || idx + 1 == pairs.size())) {
            // Calculate the percentage of completion
            double progress = static_cast<double>(idx + 1) / static_cast<double>(pairs.size()) * 100.0;
            // Critical section to ensure thread-safe progress reporting
            #pragma omp critical(progress)
            // Print progress information with formatted percentage (1 decimal place)
            std::cout << "Progress: " << (idx + 1) << "/" << pairs.size() << " ("
                      << std::fixed << std::setprecision(1) << progress << "%)\n";
        }
    }

    // Record the end time for performance measurement
    double end_time = omp_get_wtime();
    // Print the total time taken for preprocessing
    std::cout << "Completed preprocessing in " << (end_time - start_time) << " seconds." << std::endl;

    // Return 0 to indicate successful program execution
    return 0;
}
