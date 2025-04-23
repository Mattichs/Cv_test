#include "include/utils.h"
#include <fstream>

void printCoordinates(const cv::Rect& rectangle, std::string className, const std::string& filename) {
    cv::Point topLeft = rectangle.tl();
    cv::Point bottomRight = rectangle.br();
    // change name using parameter
    std::ofstream output("output.txt");
    output << className << " " <<  topLetf.x << " " <<  topLetf.y << " " << bottomRight.x << " " << bottomRight.y;
    MyFile.close();
}