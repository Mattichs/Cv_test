#include <opencv2/core/types.hpp>
#include <string>
struct Detection {
    cv::Rect roi;
    float prob;
    std::string className;
};


Detection getDetectionWithMaxOverlap(const std::vector<Detection>& detections, float iouThreshold);

bool compareByProb(const Detection &a, const Detection &b);

void printCoordinates(const std::vector<Detection>& detection, const std::string& filename);