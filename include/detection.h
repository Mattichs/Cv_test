#ifndef _DETECTION_
#define _DETECTION_

#include "opencv2/imgproc.hpp"
#include <opencv2/ml.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <utils.h>

/**
 * @brief  This file contains methods that are usefull for detetecting the images
 * 
 */

/**
 * @brief Computes the histogram of visual words for a given set of descriptors.
 * 
 * This function takes a set of SIFT descriptors and a vocabulary (a set of visual words)
 * and computes a histogram representing the frequency of each visual word in the descriptors.
 *
 * @param descriptors The input descriptors (SIFT features) extracted from an image patch.
 * @param vocabulary The vocabulary (a matrix where each row represents a visual word).
 * @return A cv::Mat representing the histogram of visual words.
 */
cv::Mat computeHistogram(const cv::Mat& descriptors, const cv::Mat& vocabulary);

/**
 * @brief Implements a sliding window object detection using a Random Forest classifier.
 *
 * This function takes an image, a trained Random Forest classifier, a vocabulary, and other parameters
 * to perform object detection using a sliding window approach. It extracts SIFT features from each window,
 * computes a histogram of visual words, and uses the Random Forest to classify the window.
 *
 * @param img The input image.
 * @param rf A pointer to the trained Random Forest classifier.
 * @param vocab The vocabulary (a matrix where each row represents a visual word).
 * @param threshold The probability threshold for accepting a detection.
 * @param winWidth The width of the sliding window.
 * @param winHeight The height of the sliding window.
 * @param stepSize The step size of the sliding window.
 * @return A vector of Detection objects representing the detected objects.
 */
std::vector<Detection> slidingWindow(const cv::Mat& img, cv::Ptr<cv::ml::RTrees> rf, const cv::Mat& vocab, float threshold = 0.4, int winWidth = 256, int winHeight = 256, int stepSize = 16);

/**
 * @brief Selects the best detection from a vector of detections based on maximum overlap.
 *
 * This function takes a vector of Detection objects and selects the one with the maximum overlap
 * with other detections, using a specified Intersection over Union (IoU) threshold.
 *
 * @param best A reference to the Detection object that will store the best detection.
 * @param detections A vector of Detection objects.
 * @param maxDetections The maximum number of top detections to consider.
 * @param iouThreshold The IoU threshold for determining overlap.
 */
void getBestDetection(Detection& best, std::vector<Detection>& detections, int maxDetections = 10, float iouThreshold = 0.3f);

/**
 * @brief Performs object detection on test images using a trained Random Forest classifier.
 *
 * This function iterates through the images in a specified test directory, performs object detection
 * using the slidingWindow function, and visualizes the results.
 *
 * @param testPath The path to the directory containing the test images.
 * @param rf A pointer to the trained Random Forest classifier.
 * @param vocab The vocabulary (a matrix where each row represents a visual word).
 */
void computeTestImages(const std::string testPath, cv::Ptr<cv::ml::RTrees> rf, const cv::Mat& vocab);

#endif