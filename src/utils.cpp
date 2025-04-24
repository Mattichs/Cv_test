#include "utils.h"
#include <fstream>
#include <iostream>

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