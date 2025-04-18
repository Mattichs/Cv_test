#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <opencv2/objdetect.hpp>
#include "opencv2/imgproc.hpp"
#include <filesystem>
#include <string>
#include <opencv2/ml.hpp>


/*  
Allora vorrei tenere il codice come è ora, addestrerò un classificatore per oggetto o posso addestrare adaboost per rilevare sfondo, 1,2,3 (dove i numeri sono nomi degli oggetti), in caso vorrei tenere conto di tutti i rettangoli che rilevano un oggetto. Per poi mantenerne solo uno, quindi penso usare NMS.
*/

using namespace cv;
using namespace std;

float computeIoU(const Rect& a, const Rect& b) {
    int intersectionArea = (a & b).area();
    int unionArea = a.area() + b.area() - intersectionArea;
    if (unionArea == 0) return 0.0;
    return (float)intersectionArea / unionArea;
}


int main() {
    //Mat img  = imread("drill.png");
    
    // lettura positives
    vector<Mat> posImg;
    Mat img;
    int target_width = 128;
    int target_height = 128;

    // leggo data positivi (dove c'è effettivamente l'oggetto)
    std::string posPath = "positives";
    for (const auto & entry : filesystem::directory_iterator(posPath)) {
        img = imread(entry.path().string());
        resize(img, img, Size(target_width, target_height)); // resize 128x128 (HOG standard)
        cvtColor(img, img, COLOR_BGR2GRAY);
        posImg.push_back(img);
    }

    // leggo dati negativi (dove non c'è l'oggetto)
    vector<Mat> negImg;
    std::string negPath = "negatives";
    for (const auto & entry : filesystem::directory_iterator(negPath)) {
        img = imread(entry.path().string());
        if(img.empty()) {
            cerr << "Non riesco a leggere il file: " << entry.path() << endl;
            continue;
        }
        resize(img, img, Size(target_width, target_height)); // resize 128x128 (HOG standard)
        cvtColor(img, img, COLOR_BGR2GRAY);
        negImg.push_back(img);
    }
    

    // HOG feature extraction
    HOGDescriptor hog;
    vector<float> descriptors;

    Mat x; // matrice delle features, ogni riga corrisponde ad una immagine

    // positive Y = 1
    Mat y = Mat::ones(posImg.size(), 1, CV_32S); // matrice delle label, per le positive una per riga valore 1 ovvero c'è un oggetto

    // negative Y = 0
    Mat y_neg = Mat::zeros(negImg.size(), 1, CV_32S);
    
    vconcat(y, y_neg, y); // creo unica matrice y (ora ha sia 0 che 1)
    
    // positive data
    for(int i = 0; i < posImg.size(); i++) {
        hog.compute(posImg[i], descriptors, Size(8,8), Size(0,0));
        Mat descMat(descriptors);
        Mat descRow = descMat.t();
        x.push_back(descRow);
    }
    // neagtive data 
    for(int i = 0; i < negImg.size(); i++) {
        hog.compute(negImg[i], descriptors, Size(8,8), Size(0,0));
        Mat descMat(descriptors);
        Mat descRow = descMat.t();
        x.push_back(descRow);
    }
    x.convertTo(x, CV_32F);


    cout << "X size: " << x.rows << " x " << x.cols << endl;
    cout << "y size: " << y.rows << " x " << y.cols << endl;

    
    // train ADABOOST
    Ptr<ml::Boost> boost = ml::Boost::create();

    boost->setBoostType(ml::Boost::REAL);
    boost->setWeakCount(300);
    boost->setWeightTrimRate(0.95);
    boost->setMaxDepth(5);
    boost->setUseSurrogates(false);

    Ptr<ml::TrainData> trainData = ml::TrainData::create(x, ml::ROW_SAMPLE, y);

    cout << "start" << endl;
    boost->train(trainData);
    cout << "end" << endl;

    boost->save("first_model.xml");


    float scaleFactor = 0.9; // Quanto riduco l'immagine a ogni step (0.9 = -10%)
    float maxScaleFactor = 1.0; // Scala iniziale 
    float minScaleFactor = 0.5; // Scala minima (ad esempio, 0.8 = -50%)

    int minSize = 128;       // non scendere sotto questa dimensione
    int windowDim = 128;
    int stride = 32;

    Mat test;
    string testPath = "testFiles/";
    for (const auto & entry : filesystem::directory_iterator(testPath)) {
        test = imread(entry.path().string());
        
        Mat original = test.clone(); // per lo scaling, ma forse posso toglierla

        Mat display = original.clone(); // per visualizzare i risultati

        vector<Rect> detectedBoxes;
        vector<float> detectedProbs;

        while (test.cols >= minSize && test.rows >= minSize && (float)test.cols / original.cols >= minScaleFactor) {
            for (int y = 0; y <= test.rows - windowDim; y += stride) {
                for (int x = 0; x <= test.cols - windowDim; x += stride) {
                    Rect roi(x, y, windowDim, windowDim);
                    Mat cropImg = test(roi).clone();  
                    cvtColor(cropImg, cropImg, COLOR_BGR2GRAY);
                    vector<float> tDescriptors;
                    hog.compute(cropImg, tDescriptors, Size(8,8), Size(0,0));
                    Mat descRow(tDescriptors);
                    descRow = descRow.t();
                    descRow.convertTo(descRow, CV_32F);
                    
                    //float response = boost->predict(descRow);
                    float rawScore = boost->predict(descRow, noArray(), ml::StatModel::RAW_OUTPUT);
                    
                    // Calcola la probabilità
                    float probability = 1.0 / (1.0 + exp(-rawScore));
                    float backgroundProbability = 1.0 - probability;
                    // Mostra i risultati
                    if (probability > 0.5f) { // Soglia per considerare un oggetto rilevato
                        float scale = (float)original.cols / test.cols;
                        Rect scaledROI(x * scale, y * scale, windowDim * scale, windowDim * scale);
                        imshow("Detections", cropImg);
                        waitKey(0); 
                        detectedBoxes.push_back(scaledROI);
                        detectedProbs.push_back(probability);
                    }
                }
            }
            resize(test, test, Size(), scaleFactor, scaleFactor); // scala immagine
        
        }

        float iouThreshold = 0.1;
        vector<Rect> filteredBoxes;
        vector<float> filteredProbs;

        // scarto rilevazioni con bassa iou 
        for (int i = 0; i < detectedBoxes.size(); i++) {
            int countOverlap = 0;
            for (int j = 0; j < detectedBoxes.size(); j++) {
                if (i == j) continue;
                if (computeIoU(detectedBoxes[i], detectedBoxes[j]) > iouThreshold) {
                    countOverlap++;
                }
            }
            if (countOverlap > 0) {
                filteredBoxes.push_back(detectedBoxes[i]);
                filteredProbs.push_back(detectedProbs[i]);
            }
        }


        int bestIdx = -1;
        float bestScore = -1;

        // prendo il boxes migliore, ovvero quello con più intersezioni con gli altri
        for (int i = 0; i < filteredBoxes.size(); i++) {
            float sumIoU = 0;
            for (int j = 0; j < filteredBoxes.size(); j++) {
                if (i == j) continue;
                sumIoU += computeIoU(filteredBoxes[i], filteredBoxes[j]);
            }
            if (sumIoU > bestScore) {
                bestScore = sumIoU;
                bestIdx = i;
            }
        }

        
        if (bestIdx >= 0) {
            rectangle(display, filteredBoxes[bestIdx], Scalar(0, 0, 255), 3);
            cout << "Rettangolo finale selezionato tra " << filteredBoxes.size() << " validi." << endl;
            imshow("Best Detection", display);
            waitKey(0);
        } else {
            cout << "Nessun rettangolo sufficientemente supportato da altri." << endl;
        }
    

    }

    

    return 0;
}

