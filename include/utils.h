#ifndef _UTILS_
#define _UTILS_
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include "opencv2/highgui.hpp"
#include <map>
#include <string>
struct Detection {
    cv::Rect roi;
    float prob;
    std::string className;
    std::string classId;
    cv::Scalar color;
};


struct LabeledBox {
    std::string classId;
    cv::Rect box;
};


cv::Mat extractSIFT(const cv::Mat& image, const cv::Mat& mask = cv::Mat());

void loadImagesAndGetFeatures(
    const std::string& datasetPath, 
    const std::vector<std::string>& classFolders, 
    std::vector<cv::Mat>& descriptors, 
    std::vector<int>& labels, 
    std::map<std::string, int>& classToLabel);

Detection getDetectionWithMaxOverlap(const std::vector<Detection>& detections, float iouThreshold);

bool compareByProb(const Detection &a, const Detection &b);

void printCoordinates(const std::vector<Detection>& detection, const std::string& filename);

void showAndSaveImageWithDetections(const cv::Mat& img, const std::vector<Detection>& detections, const std::string& filename);

float IoU(const cv::Rect& a, const cv::Rect& b);

std::vector<std::string> splitString(const std::string& str);

std::vector<LabeledBox> read_boxes(const std::string& path);

void calcAvgIOU(const std::string& predictedLabelsPath);


#endif