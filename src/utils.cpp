#include "../include/utils.h"
#include <fstream>
#include <iostream>
#include <opencv2/xfeatures2d.hpp>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <set>
#include <sstream>

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

void clearOutputFile(const std::string& filename){

}

void printCoordinates(const std::vector<Detection>& detections, const std::string& filename) {
    // creo il file 
    std::ofstream output(filename);
    for(const auto& d : detections) {
        if(!d.roi.empty()) {
            cv::Point topLeft = d.roi.tl();
            cv::Point bottomRight = d.roi.br();
            output << d.classId << "_" << d.className << " " << topLeft.x << " " << topLeft.y << " " << bottomRight.x << " " << bottomRight.y << std::endl;
        }
    }
    output.close();
    
}

void showAndSaveImageWithDetections(const cv::Mat& img, const std::vector<Detection>& detections, const std::string& filename){
    for(const auto& d : detections) {
        cv::rectangle(img, d.roi, d.color, 2);
        putText(img, d.className, cv::Point(d.roi.x, d.roi.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, d.color, 2);
    }
    cv::imwrite(filename, img);
    /*  cv::imshow("Detections", img);
    cv::waitKey(0); */
}

float IoU(const cv::Rect& a, const cv::Rect& b) {
    float interArea = (a & b).area();
    float unionArea = (a | b).area();
    if (unionArea == 0) return 0.0f;
    return interArea / unionArea;
}

std::vector<std::string> splitString(const std::string& str) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;

    while (ss >> token) {
        tokens.push_back(token);
    }

    return tokens;
}

std::vector<LabeledBox> read_boxes(const std::string& path) {
    std::ifstream file(path);
    std::vector<LabeledBox> boxes;
    int cls, x1, y1, x2, y2;
    std::string line;
    std::vector<std::string> words;
    while (getline (file, line)) {
        //std::cout << line << std::endl;
        words = splitString(line);
        //std::cout << words[0] << std::endl;
        boxes.push_back({words[0], cv::Rect(cv::Point(std::stoi(words[1]), std::stoi(words[2])), cv::Point(std::stoi(words[3]), std::stoi(words[4])))});
    }
    return boxes;
}

void calcAvgIOU(const std::string& predictedLabelsPath) {
    std::map<std::string, float> iouPerClass= {
        {"006_mustard_bottle", 0},
        {"035_power_drill", 0},
        {"004_sugar_box", 0}
    };
    std::map<std::string, int> matchesPerClass;
    std::map<std::string, int> tpClasses = {
        {"006_mustard_bottle", 0},
        {"035_power_drill", 0},
        {"004_sugar_box", 0}
    };
    std::string trueLabelPath = "../ground_truth_labels/";
    for(const auto & entry : std::filesystem::directory_iterator(predictedLabelsPath)) {
        std::string filename = entry.path().filename();
        std::string gt_path = trueLabelPath + filename;
        std::string pred_path = predictedLabelsPath + filename;
        
        // just for debug
        if (!std::filesystem::exists(pred_path)) {
            std::cerr << "Error: Predicted file not found: " << pred_path << std::endl;
            break;    
        }
        if (!std::filesystem::exists(gt_path)) {
            std::cerr << "Error: Ground truth file not found: " << gt_path << std::endl;
            break;
        }

       /*  std::cout << pred_path << std::endl;
        std::cout << gt_path << std::endl; */
        
        std::vector<LabeledBox> predBoxes = read_boxes(pred_path);
        std::vector<LabeledBox> truthBoxes = read_boxes(gt_path);
        
        /* std::cout << predBoxes.size() << std::endl;
        std::cout << truthBoxes.size() << std::endl;
         */
        for(const auto & gt : truthBoxes) {
            std::cout << gt.classId << std::endl;
            for(const auto& pred : predBoxes) {
                if(pred.classId == gt.classId) {
                    std::cout << pred.classId << std::endl;
                    float iou = IoU(gt.box, pred.box);
                    std::cout << iou << std::endl;
                    iouPerClass[gt.classId] += iou;
                    matchesPerClass[gt.classId]++;
                    if(iou > 0.5) tpClasses[gt.classId]++;
                }
            }
        }
        std::cout << filename << std::endl;
    }
    for (const auto& [classId, tp] : tpClasses) {
        std::cout << "TP " << classId << ": " << tp << std::endl;
    }
    

    for (const auto& [classId, totalIoU] : iouPerClass) {
        float miou = totalIoU / matchesPerClass[classId];
        std::cout << "mIOU " << classId << ": " << miou << std::endl;
    }
    
}
