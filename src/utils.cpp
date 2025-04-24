#include "utils.h"
#include <fstream>
#include <iostream>
#include <opencv2/xfeatures2d.hpp>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

cv::Mat extractSIFT(const cv::Mat& image, const cv::Mat& mask) {
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    
    if (!mask.empty()) {
        sift->detectAndCompute(image, mask, keypoints, descriptors);
    } else {
        sift->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
    }
    return descriptors;
}

void loadImagesAndGetFeatures(const std::string& datasetPath, const std::vector<std::string>& classFolders, std::vector<cv::Mat>& descriptors, std::vector<int>& labels, std::map<std::string, int>& classToLabel) {
    for(const auto& folder : classFolders) {
        std::string currentFolder = datasetPath + folder + "/";
        for(const auto& entry : std::filesystem::directory_iterator(currentFolder)) {
            std::string filename = entry.path().filename().string();
            if (filename.find("_color.png") == std::string::npos) {
                continue;
            }
            std::string base = filename.substr(0, filename.find("_color.png"));
            std::string colorPath = currentFolder + base + "_color.png";
            std::string maskPath = currentFolder + base + "_mask.png";

            cv::Mat color = cv::imread(colorPath);
            if(color.empty()) {
                std::cerr << "Non riesco a leggere l'immagine: " << colorPath << std::endl;
                continue;
            }
            cv::Mat mask = cv::imread(maskPath, cv::IMREAD_GRAYSCALE);
            if(mask.empty()) {
                std::cerr << "Non riesco a leggere la maschera: " << maskPath << std::endl;
                continue;
            }
            cv::Mat gray;
            cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
            cv::Mat claheResult;
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
            clahe->setClipLimit(4.0);
            clahe->apply(gray, claheResult);

            cv::Mat desc = extractSIFT(claheResult, mask);
            if (!desc.empty()) {
                
                descriptors.push_back(desc);
                labels.push_back(classToLabel[folder]);  
            }    
        }
    }
}


Detection getDetectionWithMaxOverlap(const std::vector<Detection>& detections, float iouThreshold) {
    if (detections.empty()) {
        throw std::invalid_argument("No detections to process");
    }

    int maxOverlapCount = 0;    // Numero di sovrapposizioni più alto trovato
    Detection bestDetection;    // La detection con il massimo numero di sovrapposizioni

    for (size_t i = 0; i < detections.size(); ++i) {
        const Detection& current = detections[i];
        int overlapCount = 0;

        // Controlla quante altre detections si sovrappongono con questa
        for (size_t j = 0; j < detections.size(); ++j) {
            if (i == j) continue;  // Non confrontare la detection con sé stessa

            const Detection& other = detections[j];
            float iou = (current.roi & other.roi).area() /
                        float((current.roi | other.roi).area());

            if (iou > iouThreshold) {
                overlapCount++;
            }
        }

        // Se questa detection ha più sovrapposizioni di altre, aggiornala come la migliore
        if (overlapCount > maxOverlapCount) {
            maxOverlapCount = overlapCount;
            bestDetection = current;
        }
    }

    return bestDetection;
}


bool compareByProb(const Detection &a, const Detection &b) {
    return a.prob > b.prob;
}

void printCoordinates(const std::vector<Detection>& detections, const std::string& filename) {
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
