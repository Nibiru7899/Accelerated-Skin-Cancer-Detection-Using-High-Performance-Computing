#include <iostream>
#include <vector>
#include <string>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <omp.h>

std::vector<std::string> get_files(const std::string& dir, const std::string& suffix) {
    std::vector<std::string> files;
    DIR* dp = opendir(dir.c_str());
    struct dirent* entry;
    while ((entry = readdir(dp)) != NULL) {
        std::string name = entry->d_name;
        if (name.size() > suffix.size() && name.substr(name.size() - suffix.size()) == suffix)
            files.push_back(dir + "/" + name);
    }
    closedir(dp);
    return files;
}

cv::Mat remove_hair(const cv::Mat& img) {
    cv::Mat gray, blackhat, mask, result;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(17, 17));
    cv::morphologyEx(gray, blackhat, cv::MORPH_BLACKHAT, kernel);
    cv::threshold(blackhat, mask, 10, 255, cv::THRESH_BINARY);
    cv::inpaint(img, mask, result, 1, cv::INPAINT_TELEA);
    return result;
}

void preprocess_folder(const std::string& input_dir, const std::string& output_dir, const std::string& suffix) {
    auto files = get_files(input_dir, suffix);
    #pragma omp parallel for
    for (int i = 0; i < files.size(); ++i) {
        cv::Mat img = cv::imread(files[i]);
        if (img.empty()) continue;

        if (suffix == "_Orig.jpg") img = remove_hair(img); // only for originals

        cv::resize(img, img, cv::Size(256, 256));
        std::string out_file = output_dir + "/" + files[i].substr(files[i].find_last_of('/') + 1);
        cv::imwrite(out_file, img);

        #pragma omp critical
        std::cout << "Processed: " << files[i] << std::endl;
    }
}

int main() {
    std::vector<std::string> classes = {"Benign", "Malignant"};
    for (const auto& c : classes) {
        std::string in_dir = "C:\\SEM-5\\Big Data Analytics\\Projects\\Dataset\\dataset\\Benign\\" + c;
        std::string out_dir = "C:\\SEM-5\\Big Data Analytics\\Projects\\Dataset\\dataset\\Malignant\\" + c;
        system(("mkdir -p " + out_dir).c_str()); // for unix-like systems
        preprocess_folder(in_dir, out_dir, "_Orig.jpg");
        preprocess_folder(in_dir, out_dir, "_GT.jpg"); // masks (skip remove_hair)
    }
    return 0;
}
