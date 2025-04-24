#include <opencv2/opencv.hpp>
#include <utils.h>

std::vector<Detection> slidingWindow(const cv::Mat& img, cv::Ptr<cv::ml::RTrees> rf, const cv::Mat& vocab, float threshold = 0.4, int winWidth = 256, int winHeight = 256, int stepSize = 16);

cv::Mat extractSIFT(const cv::Mat& image, const cv::Mat& mask = cv::Mat());

// Funzione per calcolare l'istogramma BoW (esempio basato sul tuo codice)
cv::Mat computeHistogram(const cv::Mat& descriptors, const cv::Mat& vocabulary);