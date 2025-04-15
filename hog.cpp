#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <opencv2/objdetect.hpp>
#include "opencv2/imgproc.hpp"
#include <filesystem>
#include <string>
#include <opencv2/ml.hpp>


using namespace cv;
using namespace std;


int main() {
    //Mat img  = imread("drill.png");
    
    // lettura positives
    vector<Mat> posImg;
    Mat img;
    int target_width = 128;
    int target_height = 128;

    // leggo data positivi (dove c'è effettivamente l'oggetto)
    std::string posPath = "data/positives";
    for (const auto & entry : filesystem::directory_iterator(posPath)) {
        img = imread(entry.path().string());
        resize(img, img, Size(target_width, target_height)); // resize 128x128 (HOG standard)
        cvtColor(img, img, COLOR_BGR2GRAY);
        posImg.push_back(img);
    }

    // leggo dati negativi (dove non c'è l'oggetto)
    vector<Mat> negImg;
    std::string negPath = "data/negatives";
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
    /* // show img
    for(int i = 0; i < negImg.size(); i++) {
        imshow("img", negImg[i]);
        waitKey(0);
    }  */

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
    boost->setWeakCount(100);
    boost->setWeightTrimRate(0.95);
    boost->setMaxDepth(1);
    boost->setUseSurrogates(false);

    Ptr<ml::TrainData> trainData = ml::TrainData::create(x, ml::ROW_SAMPLE, y);

    boost->train(trainData);


    boost->save("first_model.xml");

    
    // test con immagine sintetica
    /* Mat test;
    std::string testPath = "drill.png";
    test = imread(testPath);
    resize(test, test, cv::Size(128, 128)); // usa la stessa risoluzione usata per il training
    cvtColor(test, test, cv::COLOR_BGR2GRAY);
    vector<float> tDescriptors;
    hog.compute(test, tDescriptors, Size(8,8), Size(0,0));
    Mat descRow(tDescriptors);
    descRow = descRow.t();

    descRow.convertTo(descRow, CV_32F);

    float response = boost->predict(descRow);
    if (response == 1.0f) {
        cout << "Oggetto rilevato" << endl;
    } else {
        cout << "Nessuna rilevazione" << endl;
    } */

    

    int windowDim = 128;
    int stride = 10;
    Mat test = imread("test.jpg");
    Mat display = test.clone();
    for(int y = 0; y < test.rows - windowDim; y+=stride) {
        for(int x = 0; x < test.cols - windowDim; x+=stride) {
            Rect roi(x, y, windowDim, windowDim);
            Mat cropImg = test(roi).clone();  
            cvtColor(cropImg, cropImg, cv::COLOR_BGR2GRAY);
            vector<float> tDescriptors;
            hog.compute(cropImg, tDescriptors, Size(8,8), Size(0,0));
            Mat descRow(tDescriptors);
            descRow = descRow.t();

            descRow.convertTo(descRow, CV_32F);

            float response = boost->predict(descRow);
            if (response == 1.0f) {
                cout << "Oggetto rilevato" << endl;
                // disegno rettangoli
                rectangle(display, roi, Scalar(0, 255,0), 2);
                imshow("img crop", cropImg);
                waitKey(0);
            } else {
                cout << "Nessuna rilevazione" << endl;
            }  
            
        } 
    }
       
    imshow("img", display);
    waitKey(0);    

    

    return 0;
}



