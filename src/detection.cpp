#include "../include/detection.h"
#include <filesystem>
#include <regex>

cv::Mat computeHistogram(const cv::Mat& descriptors, const cv::Mat& vocabulary) {
    if (descriptors.empty() || vocabulary.empty()) {
        // Restituisci un istogramma vuoto o di zeri se non ci sono descrittori
        // La dimensione deve corrispondere alla dimensione del vocabolario (VOCAB_SIZE)
        int vocabSize = vocabulary.rows; // Assumendo che VOCAB_SIZE sia la dimensione delle righe
         if (vocabSize <= 0) {
            std::cerr << "Errore: Dimensione del vocabolario non valida in computeHistogram." << std::endl;
            return cv::Mat(); // Restituisce Mat vuota in caso di errore grave
         }
        return cv::Mat::zeros(1, vocabSize, CV_32F);
    }
    // Assicurati che i descrittori siano CV_32F come richiesto da BFMatcher
    cv::Mat descriptorsFloat;
    if (descriptors.type() != CV_32F) {
        descriptors.convertTo(descriptorsFloat, CV_32F);
    } else {
        descriptorsFloat = descriptors;
    }

    // Usa BFMatcher per trovare la visual word più vicina per ogni descrittore
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    // Alternativa più semplice: cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create();

    cv::BOWImgDescriptorExtractor bowDE(matcher);
    bowDE.setVocabulary(vocabulary);

    cv::Mat histogram;
    
    std::vector<cv::DMatch> matches;
    matcher->match(descriptorsFloat, vocabulary, matches); // Trova la parola del vocabolario più vicina per ogni descrittore

    int vocabSize = vocabulary.rows;
    histogram = cv::Mat::zeros(1, vocabSize, CV_32F); // Istogramma inizializzato a zero

    for (const auto& match : matches) {
        if (match.queryIdx < descriptorsFloat.rows && match.trainIdx < vocabSize) {
            histogram.at<float>(0, match.trainIdx)++; // Incrementa il bin corrispondente alla visual word
        }
    }

    // Normalizzazione L1 (opzionale ma comune per BoW)
    cv::normalize(histogram, histogram, 1.0, 0.0, cv::NORM_L1);

    return histogram;
}


std::vector<Detection> slidingWindow(const cv::Mat& img, cv::Ptr<cv::ml::RTrees> rf, const cv::Mat& vocab, float threshold, int winWidth, int winHeight, int stepSize) {
    std::vector<Detection> detections;
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    for (int y = 0; y <= img.rows - winHeight; y += stepSize) {
        for (int x = 0; x <= img.cols - winWidth; x += stepSize) {
            cv::Rect roi(x, y, winWidth, winHeight);
            cv::Mat patchColor = img(roi);
            cv::Mat patchGray;
            cv::cvtColor(patchColor, patchGray, cv::COLOR_BGR2GRAY);

            cv::Mat patchClaheResult;
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
            clahe->setClipLimit(4.0); // Usa lo stesso limite del training
            clahe->apply(patchGray, patchClaheResult);
            
            // estraggo descrittori SIFT
            cv::Mat patchDescriptors = extractSIFT(patchClaheResult);

            cv::Mat testHistogram = computeHistogram(patchDescriptors, vocab);

            cv::Mat predictionResults;
            rf->getVotes(testHistogram, predictionResults, 0);
            //cout << predictionResults << endl;
            std::vector<std::string> classNames = {"mustard_bottle", "power_drill", "sugar_box"};
            std::vector<std::string> classIds = {"006", "035", "004"};
            std::vector<cv::Scalar> classColors = {cv::Scalar (255,0,0), cv::Scalar (0,255,0), cv::Scalar (0,0,255)};
            std::vector<double> votesPerClass(classNames.size());
            cv::Mat votes = predictionResults.row(1);
            //cout << votes << endl;
            float sum = 0.0f;
            for(int i = 0; i < votes.cols; i++) sum+=votes.at<int>(0,i);
            
            for (int i= 0; i < votes.cols; i++) {
                float prob = votes.at<int>(0,i)/sum;
                //std::cout << classNames[i] << " : " << prob << std::endl;    
                if(prob > threshold) {
                    Detection temp;
                    temp.className = classNames[i];
                    temp.classId = classIds[i];
                    temp.roi = roi;
                    temp.prob = prob;
                    temp.color = classColors[i];
                    detections.push_back(temp);
                }
            }
        
            /* imshow("patch", patchColor);      
            waitKey(0);    */       
        }
    }
    
    
    return detections;
}


void getBestDetection(Detection& best, std::vector<Detection>& detections, int maxDetections, float iouThreshold) {
     if(!detections.empty()) {
        if(detections.size() == 1) {
            //cv::rectangle(testColor, detections[0].roi, cv::Scalar(0, 0, 255), 2);
            best = detections[0];
        } else {
            std::sort(detections.begin(), detections.end(), compareByProb);
        
            std::vector<Detection> topDetection;
        
            if(detections.size() < maxDetections) maxDetections = detections.size();
            for(int i = 0; i < maxDetections;i++) topDetection.push_back(detections[i]);
            
            // Soglia per determinare se due box sono considerate sovrapposte
            best = getDetectionWithMaxOverlap(topDetection, iouThreshold);
            //cv::rectangle(testColor, best.roi, cv::Scalar(0, 0, 255), 2);
        }
    } 
}


void computeTestImages(const std::string testPath, cv::Ptr<cv::ml::RTrees> rf, const cv::Mat& vocab) {
    std::string resPath="../results/";
    for (const auto& entry : std::filesystem::directory_iterator(testPath)) {
        // --- Caricamento e Preprocessing Immagine di Test ---
        cv::Mat testColor = cv::imread(entry.path().string());
        if (testColor.empty()) {
            std::cerr << "Errore: Impossibile leggere l'immagine di test: " << std::endl;
            return;
        }

        std::vector<Detection> detections = slidingWindow(testColor, rf, vocab);
        //std::cout << detections.size() << std::endl;
        if(detections.size() == 0) {
            std::cout << "No detections sorry :(" << std::endl;
        }
        
        // split detection to mustard, drill and sugar
        std::vector<Detection> mustardDetections;
        std::vector<Detection> drillDetections;
        std::vector<Detection> sugarDetections;
        for(const auto & d: detections) {
            if(d.className == "mustard") {
                mustardDetections.push_back(d);
            } else if(d.className == "drill") {
                drillDetections.push_back(d);
            } else {
                sugarDetections.push_back(d);
            }
        }
        // These are for debugs
        //std::cout << "mustard Detections:" << mustardDetections.size() << std::endl;
        //std::cout << "drill Detections:" << drillDetections.size() << std::endl;
        //std::cout << "sugar Detections:" << sugarDetections.size() << std::endl;
        
        Detection bestMustard;
        getBestDetection(bestMustard, mustardDetections);

        Detection bestDrill;
        getBestDetection(bestDrill, drillDetections);

        Detection bestSugar;
        getBestDetection(bestSugar, sugarDetections);

        std::vector<Detection> bestDetections = {bestMustard, bestDrill, bestSugar};
        std::regex pattern("color.jpg", std::regex_constants::icase);
        std::string filename= resPath + "/labels/" + std::regex_replace(entry.path().filename().string(), pattern, "box.txt");
        printCoordinates(bestDetections, filename);
        filename= resPath + "/images/" + entry.path().filename().string();
        showAndSaveImageWithDetections(testColor, bestDetections, filename);
    }
}