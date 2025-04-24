#include "utils.h"
#include <fstream>
#include <iostream>

void printCoordinates(const std::vector<Detection>& detections, std::string className, const std::string& filename) {
    // creo il file 
    std::ofstream output(filename);
    for(const auto& d : detections) {
        cv::Point topLeft = d.roi.tl();
        cv::Point bottomRight = d.roi.br();
        std::cout << d.className << " " << topLeft.x << " " << topLeft.y << " " << bottomRight.x << " " << bottomRight.y << std::endl;
        // change name using parameter
       
        //output << className << " " <<  topLetf.x << " " <<  topLetf.y << " " << bottomRight.x << " " << bottomRight.y;
        //MyFile.close();
    }
    
}