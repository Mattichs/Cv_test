#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>
#include <string>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

// load images 
void loadImages(vector<Mat>& imgVec, string path) {
    Mat img;
    for (const auto & entry : filesystem::directory_iterator(path)) {
        img = imread(entry.path().string());
        if(img.empty()) {
            cerr << "Non riesco a leggere il file: " << entry.path() << endl;
            continue;
        }
        cvtColor(img, img, COLOR_BGR2GRAY);
        imgVec.push_back(img);
    }
}

// intersect over union
float iou(const vector<Point>& truth, const vector<Point>& predict) {
    int a = max(truth[0].x, predict[0].x);
    int b = max(truth[0].y, predict[0].y);
    int c = min(truth[1].x, predict[1].x);
    int d = min(truth[1].y, predict[1].y);
    if (c <= a || d <= b) return 0.0f;
    int intersection = (c-a) * (d-b);
    int areaTruth = (truth[1].x - truth[0].x)*(truth[1].y - truth[0].y);
    int areaPredict = (predict[1].x - predict[0].x)*(predict[1].y - predict[0].y);;
    int unionArea = areaTruth + areaPredict - intersection;

    if(unionArea <= 0 ) return 0.0f;

    float conf = static_cast<float> (intersection) / unionArea;
        
    return conf;
} 

vector<string> split(const string& str) {
    vector<string> tokens;
    istringstream iss(str);
    string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}


// Struttura per memorizzare keypoint e descrittori associati a ogni vista
struct ViewData {
    Mat image;
    vector<KeyPoint> keypoints;
    Mat descriptors;
};

int main( int argc, char* argv[] ) {
    
    Mat testImg = imread(  "../testFiles/test.jpg", IMREAD_GRAYSCALE );
    if ( testImg.empty() ){
        cout << "Could not open or find the image!\n" << endl;
        return -1;
    }

    // leggo data positivi (dove c'è effettivamente l'oggetto)
    std::string posPath = "../data/positives";
    vector<Mat> posImg;
    loadImages(posImg, posPath);

    Ptr<SIFT> detector = SIFT::create();

    // computazione SIFT feature delle immagini sintetiche
    
    vector<ViewData> views;
    
    for (const auto& img : posImg) {
        ViewData data;
        data.image = img.clone(); // salva anche immagine (utile per visualizzazione)
        detector->detectAndCompute(img, noArray(), data.keypoints, data.descriptors);
        views.push_back(data);
    }

    // sliding windows


    Mat outputImage;
    cvtColor(testImg, outputImage, COLOR_GRAY2BGR);
    //int windowSize = 256;
    int stride = 32;
    vector<pair<int, int>> windows = {{128, 256}, {256, 128}};

    // read ground truth
    // collect ground truth
    ifstream ReadFile("../testFiles/test.txt");
    string text;
    getline(ReadFile, text);
    vector<string> words = split(text);

    vector<Point> bb_truth;
    bb_truth.push_back(Point(stoi(words[1]), stoi(words[2])));
    bb_truth.push_back(Point(stoi(words[3]), stoi(words[4])));


    for(const auto& window : windows) {
        for(int y = 0; y < testImg.rows - window.second; y+=stride) {
            for(int x = 0; x < testImg.cols - window.first; x+=stride) {
                Rect roi(x, y, window.first, window.second);
                Mat cropImg = testImg(roi).clone();

                vector<KeyPoint> windowKp;
                Mat windowDesc;
                detector->detectAndCompute( cropImg, noArray(), windowKp, windowDesc);
                if(windowDesc.empty()) continue;
                
                // matching 
                for(const auto & view : views) {
                    
                    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
                    // Since SURF is a floating-point descriptor NORM_L2 is used
                    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
                    std::vector< std::vector<DMatch> > knn_matches;
                    matcher->knnMatch( view.descriptors, windowDesc, knn_matches, 2 );
                    //-- Filter matches using the Lowe's ratio test
                    const float ratio_thresh = 0.7f;
                    std::vector<DMatch> good_matches;
                    for (size_t i = 0; i < knn_matches.size(); i++) {
                        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
                            good_matches.push_back(knn_matches[i][0]);
                        }
                    }
                    if(good_matches.size() > 5) {
                        vector<Point> bb_detected; 
                        bb_detected.push_back(Point(x, y));
                        bb_detected.push_back(Point(x+ window.first, y+ window.second));
                        float conf = iou(bb_truth, bb_detected);
                        
                        if(conf > 0.8) {
                            cout << "Confidence: " << conf << endl;
                            rectangle(outputImage, roi, Scalar(0, 255, 0), 2);
                            //imshow("output", cropImg);
                            //waitKey(0);
                            // break if you want only one detection per window
                            break;
                        }
                    }
                    
                    
                }
            }
        }
    }
    imshow("output", outputImage);
    waitKey(0);

    return 0;
}