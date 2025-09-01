// Parallel image preprocessing (resize) using OpenMP + OpenCV
// Usage:
//   preprocess <input_root> <output_root> [width height] [num_threads]
// Example:
//   ./preprocess_openmp dataset preprocessed 224 224 4

// Include OpenCV library for image processing functions
#include <opencv2/opencv.hpp>
// Include filesystem library for directory and file operations
#include <filesystem>
// Include vector for storing file paths
#include <vector>
// Include string for string operations
#include <string>
// Include iostream for input/output operations
#include <iostream>
// Include algorithm for std::transform
#include <algorithm>
// Include OpenMP library for parallel processing
#include <omp.h>
// Include iomanip for formatting output
#include <iomanip>

// Create an alias 'fs' for the std::filesystem namespace to simplify code
namespace fs = std::filesystem;

/**
 * Function to check if a file has a valid image extension
 * @param p The file path to check
 * @return true if the file has a valid image extension, false otherwise
 */
static bool has_image_ext(const fs::path &p) {
    // Extract file extension from the path
    std::string ext = p.extension().string();
    // Convert extension to lowercase for case-insensitive comparison
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return (char)std::tolower(c); });
    // Check if extension matches any of the supported image formats
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tif" || ext == ".tiff";
}

/**
 * Function to gather all image files from a directory and its subdirectories
 * @param root The root directory to search for images
 * @param out Vector to store the paths of found image files
 */
static void gather_file_list(const std::string &root, std::vector<std::string> &out) {
    // Check if the input root directory exists
    if (!fs::exists(root)) {
        std::cerr << "Input root does not exist: " << root << "\n";
        return;
    }
    // Recursively iterate through all files in the directory and subdirectories
    for (auto const &entry : fs::recursive_directory_iterator(root)) {
        // Check if the entry is a regular file and has a valid image extension
        if (entry.is_regular_file() && has_image_ext(entry.path())) {
            // Add the file path to the output vector
            out.emplace_back(entry.path().string());
        }
    }
}

/**
 * Function to process a single image: read, resize, and save to output directory
 * @param path Path to the input image
 * @param input_root Root directory of input images
 * @param output_root Root directory for output images
 * @param width Target width for resized image
 * @param height Target height for resized image
 * @param thread_id ID of the thread processing this image (for logging)
 */
static void process_image(const std::string& path, 
                          const std::string& input_root,
                          const std::string& output_root,
                          int width, int height, int thread_id) {
    // Read the image from disk using OpenCV
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    // Check if image was successfully loaded
    if (img.empty()) {
        std::cerr << "[Thread " << thread_id << "] Failed to read: " << path << "\n";
        return;
    }
    
    // Create a new matrix to store the resized image
    cv::Mat resized;
    // Resize the image to the target dimensions using area interpolation (best for downsampling)
    cv::resize(img, resized, cv::Size(width, height), 0, 0, cv::INTER_AREA);

    // Calculate the relative path of the image from the input root
    fs::path rel = fs::relative(path, input_root);
    // Create the output path by joining the output root with the relative path
    fs::path out_path = fs::path(output_root) / rel;
    
    // Create directories safely with critical section to avoid race conditions
    // where multiple threads might try to create the same directory simultaneously
    #pragma omp critical(dir_creation)
    {
        // Create all parent directories for the output file if they don't exist
        fs::create_directories(out_path.parent_path());
    }
    
    // Write the resized image to disk
    if (!cv::imwrite(out_path.string(), resized)) {
        // Log error if writing fails
        std::cerr << "[Thread " << thread_id << "] Failed to write: " << out_path.string() << "\n";
    }
}

/**
 * Main function - entry point of the program
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 * @return Exit code: 0 for success, non-zero for errors
 */
int main(int argc, char** argv) {
    // Default parameters for input and output directories
    std::string input_root = "dataset";
    std::string output_root = "preprocessed";
    // Default target dimensions for resized images
    int width = 224, height = 224;
    // Default to maximum available threads on the system
    int num_threads = omp_get_max_threads();

    // Parse command-line arguments if provided
    if (argc >= 3) {
        // Set input and output directories from arguments
        input_root = argv[1];
        output_root = argv[2];
    }
    if (argc >= 5) {
        // Set target width and height from arguments
        width = std::stoi(argv[3]);
        height = std::stoi(argv[4]);
    }
    if (argc >= 6) {
        // Set number of threads from arguments
        num_threads = std::stoi(argv[5]);
        // Configure OpenMP to use the specified number of threads
        omp_set_num_threads(num_threads);
    }

    // Vector to store all image file paths
    std::vector<std::string> files;
    // Populate the vector with all image files from the input directory
    gather_file_list(input_root, files);
    // Sort file paths for consistent processing order
    std::sort(files.begin(), files.end());
    
    // Display information about the processing task
    std::cout << "Discovered " << files.size() << " files under '" << input_root << "'." << std::endl;
    std::cout << "Using " << num_threads << " threads for processing." << std::endl;
    
    // Create the output directory before starting parallel processing
    fs::create_directories(output_root);

    // Record the start time for performance measurement
    double start_time = omp_get_wtime();
    
    // Start parallel region - this creates multiple threads
    #pragma omp parallel
    {
        // Get this thread's ID for logging
        int thread_id = omp_get_thread_num();
        // Get the total number of threads for potential workload calculations
        int total_threads = omp_get_num_threads();
        
        // Distribute loop iterations among threads with dynamic scheduling
        // Dynamic scheduling helps balance the workload if processing times vary
        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < files.size(); i++) {
            // Process the current image file
            process_image(files[i], input_root, output_root, width, height, thread_id);
            
            // Periodically print progress (only from thread 0 to avoid output clutter)
            if (thread_id == 0 && (i % 100 == 0 || i == files.size() - 1)) {
                // Use critical section to avoid interleaved output from multiple threads
                #pragma omp critical(cout)
                {
                    // Display progress information with formatting
                    std::cout << "Progress: " << i+1 << "/" << files.size() << " (" 
                              << std::fixed << std::setprecision(1) 
                              << (100.0 * (i+1) / files.size()) << "%)" << std::endl;
                }
            }
        }
    }
    
    // Record the end time after all processing is complete
    double end_time = omp_get_wtime();
    // Display the total execution time
    std::cout << "Completed preprocessing in " << (end_time - start_time) << " seconds." << std::endl;
    
    // Return success code
    return 0;
}
