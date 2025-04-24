#include <opencv2/core/types.hpp>
#include <string>
struct Detection {
    cv::Rect roi;
    float prob;
    std::string className;
};

void printCoordinates(const std::vector<Detection>& detection, const std::string& filename);