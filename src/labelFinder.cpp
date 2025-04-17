#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;
namespace fs = filesystem;

int main() {
    string images_dir = "../data/006_mustard_bottle/training_images/";
    string masks_dir = "../data/006_mustard_bottle/masks/";
    string output_file = "../positives.txt";

    ofstream out(output_file);
    if (!out.is_open()) {
        std::cerr << "Cannot open output file!\n";
        return -1;
    }

    for (const auto& entry : fs::directory_iterator(masks_dir)) {
        string mask_path = entry.path().string();
        string filename = entry.path().stem().string(); // es: img1
        string image_path = images_dir + filename + ".jpg";

        Mat mask = imread(mask_path, cv::IMREAD_GRAYSCALE);
        if (mask.empty()) {
            cerr << "Cannot load mask: " << mask_path << endl;
            continue;
        }

        // Assicurati che la maschera sia binaria
        threshold(mask, mask, 128, 255, cv::THRESH_BINARY);

        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        if (contours.empty()) {
            continue;
        }

        out << image_path << " " << contours.size();

        for (const auto& contour : contours) {
            Rect box = boundingRect(contour);
            out << " " << box.x << " " << box.y << " " << box.width << " " << box.height;
        }
        out << endl;
    }

    out.close();
    cout << "File positives.txt generated!" << std::endl;

    return 0;
}
