#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
using namespace std;
using namespace cv;

int main( int argc, const char** argv )
{
    CascadeClassifier mustard;
    if (!mustard.load("../cascade/cascade.xml")) {
        std::cerr << "Cannot  load cascade.xml!" << std::endl;
        return -1;
    }
    Mat image = imread("../data/006_mustard_bottle/test_images/6_0001_000121-color.jpg");
    if (image.empty()) {
        std::cerr << "Cannot find the image!" << std::endl;
        return -1;
    }
    vector<Rect> objects;
    mustard.detectMultiScale(image, objects, 1.1, 3, 0, cv::Size(24, 24));

    // Disegna i rettangoli
    for (const auto& rect : objects) {
        rectangle(image, rect, cv::Scalar(0, 255, 0), 2);
    }
    imshow("Mustard", image);
    waitKey(0);
    return 0;
}