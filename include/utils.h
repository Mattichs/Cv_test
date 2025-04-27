//@author bastianello_mattia
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

/**
 * @brief Extracts SIFT descriptors from an image.
 * @param image The image to extract descriptors from.
 * @param mask An optional mask to specify regions of interest.
 * @return The extracted SIFT descriptors.
 */
cv::Mat extractSIFT(const cv::Mat& image, const cv::Mat& mask = cv::Mat());

/**
 * @brief Loads images from a dataset and extracts features.
 * @param datasetPath The path to the dataset.
 * @param classFolders A vector of class folder names.
 * @param descriptors A vector to store the extracted descriptors.
 * @param labels A vector to store the labels.
 * @param classToLabel A map to convert class names to labels.
 */
void loadImagesAndGetFeatures(
    const std::string& datasetPath, 
    const std::vector<std::string>& classFolders, 
    std::vector<cv::Mat>& descriptors, 
    std::vector<int>& labels, 
    std::map<std::string, int>& classToLabel);

/**
 * @brief Gets the detection with the maximum overlap based on the IOU threshold.
 * @param detections A vector of detections.
 * @param iouThreshold The Intersection over Union (IOU) threshold.
 * @return The detection with the maximum overlap.
 */  
Detection getDetectionWithMaxOverlap(const std::vector<Detection>& detections, float iouThreshold);

/**
 * @brief Gets the detection with the maximum overlap based on the IOU threshold.
 * @param detections A vector of detections.
 * @param iouThreshold The Intersection over Union (IOU) threshold.
 * @return The detection with the maximum overlap.
 */ 
bool compareByProb(const Detection &a, const Detection &b);

/**
 * @brief Prints the coordinates of the detections to a file.
 * @param detection A vector of detections.
 * @param filename The name of the file to write the coordinates to.
 */
void printCoordinates(const std::vector<Detection>& detection, const std::string& filename);

/**
 * @brief Shows and saves the image with the drawn detections.
 * @param img The image to display and save.
 * @param detections A vector of detections to draw on the image.
 * @param filename The name of the file to save the image to.
 */
void showAndSaveImageWithDetections(const cv::Mat& img, const std::vector<Detection>& detections, const std::string& filename);

/**
 * @brief Calculates the Intersection over Union (IOU) of two rectangles.
 * @param a The first rectangle.
 * @param b The second rectangle.
 * @return The IOU value.
 */
float IoU(const cv::Rect& a, const cv::Rect& b);

/**
 * @brief Splits a string into a vector of strings based on spaces.
 * @param str The string to split.
 * @return A vector of strings.
 */
std::vector<std::string> splitString(const std::string& str);

/**
 * @brief Reads bounding boxes from a file.
 * @param path The path to the file.
 * @return A vector of LabeledBox objects.
 */
std::vector<LabeledBox> read_boxes(const std::string& path);

/**
 * @brief Calculates the average IOU for the predicted labels.
 * @param predictedLabelsPath The path to the predicted labels.
 */
void calcAvgIOU(const std::string& predictedLabelsPath);


#endif
