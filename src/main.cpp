#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml.hpp>
#include <filesystem>
#include <iostream>
#include <numeric>

namespace fs = std::filesystem;
using namespace cv;
using namespace cv::ml;
using namespace cv::xfeatures2d;
using namespace std;

const int VOCAB_SIZE = 100;
const int WIN_HEIGHT= 256;
const int WIN_WIDTH = 128;
const int STEP_SIZE = 32;

struct Detection {
    Rect roi;
    float prob;
};


Detection getDetectionWithMaxOverlap(const vector<Detection>& detections, float iouThreshold) {
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
// === Estrai SIFT dove la maschera è positiva ===
Mat extractSIFT(const Mat& imgGray, const Mat& mask) {
    Ptr<SIFT> sift = SIFT::create();
    vector<KeyPoint> keypoints;
    Mat descriptors;
    sift->detectAndCompute(imgGray, mask, keypoints, descriptors);
    return descriptors;
}

// === Istogramma BoVW ===
Mat computeHistogram(const Mat& descriptors, const Mat& vocabulary) {
    Mat hist = Mat::zeros(1, vocabulary.rows, CV_32F);
    if (descriptors.empty()) return hist;

    for (int i = 0; i < descriptors.rows; ++i) {
        double minDist = DBL_MAX;
        int bestIdx = 0;
        for (int j = 0; j < vocabulary.rows; ++j) {
            double dist = norm(descriptors.row(i), vocabulary.row(j));
            if (dist < minDist) {
                minDist = dist;
                bestIdx = j;
            }
        }
        hist.at<float>(0, bestIdx) += 1.0f;
    }

    return hist;
}

// === Sliding Window Detection ===
vector<Detection> slidingWindow(const Mat& img, Ptr<SVM> svm, const Mat& vocab, float threshold = 0.7) {
    vector<Detection> detections;
    Ptr<SIFT> sift = SIFT::create();

    for (int y = 0; y <= img.rows - WIN_HEIGHT; y += STEP_SIZE) {
        for (int x = 0; x <= img.cols - WIN_WIDTH; x += STEP_SIZE) {
            Rect roi(x, y, WIN_WIDTH, WIN_HEIGHT);
            Mat patchGray;
            cvtColor(img(roi), patchGray, COLOR_BGR2GRAY);

            vector<KeyPoint> kp;
            Mat desc;
            sift->detectAndCompute(patchGray, noArray(), kp, desc);

            if (desc.empty()) continue;

            Mat hist = computeHistogram(desc, vocab);
            float response = svm->predict(hist, noArray(), StatModel::RAW_OUTPUT);
            float prob = 1.0f / (1.0f + exp(-response));

            if (prob > threshold) {
                Detection temp;
                temp.roi = roi;
                temp.prob = prob;
                detections.push_back(temp);
            }
                
        }
    }
    return detections;
}

int main() {
    string dataset_path = "models/";

    vector<Mat> allDescriptors;
    vector<int> labels;

    for (const auto& entry : fs::directory_iterator(dataset_path)) {
        string filename = entry.path().filename().string();

        // qui potrei estrarre lo sofndo dai bg ma overfitta per riconoscere solo il background
        if (filename.find("_color.png") == string::npos) continue;

        string base = filename.substr(0, filename.find("_color.png"));
        string colorPath = dataset_path + base + "_color.png";
        string maskPath = dataset_path + base + "_mask.png";

        Mat color = imread(colorPath);
        Mat mask = imread(maskPath, IMREAD_GRAYSCALE);
        Mat gray;
        cvtColor(color, gray, COLOR_BGR2GRAY);

        Mat desc = extractSIFT(gray, mask);
        if (!desc.empty()) {
            allDescriptors.push_back(desc);
            labels.push_back(1);  // positivo
        }

        Mat desc_bg = extractSIFT(gray, Mat()); // maschera nulla = sfondo
        if (!desc_bg.empty()) {
            allDescriptors.push_back(desc_bg);
            labels.push_back(0);  // negativo
        }
    }

    // Unisci tutti i descrittori
    Mat descriptorsAll;
    vconcat(allDescriptors, descriptorsAll);

    // KMeans
    cout << "KMeans clustering...\n";
    Mat labelsK;
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 100, 0.01);
    BOWKMeansTrainer bowTrainer(VOCAB_SIZE, criteria, 3, KMEANS_PP_CENTERS);
    Mat vocabulary = bowTrainer.cluster(descriptorsAll);

    // Istogrammi
    Mat trainData;
    for (const auto& desc : allDescriptors) {
        Mat hist = computeHistogram(desc, vocabulary);
        trainData.push_back(hist);
    }

    // SVM
    Ptr<SVM> svm = SVM::create();
    svm->setKernel(SVM::LINEAR);
    svm->setType(SVM::C_SVC);
    svm->train(trainData, ROW_SAMPLE, labels);
    svm->save("svm_model.yml");
    FileStorage fs_vocab("vocab.yml", FileStorage::WRITE);
    fs_vocab << "vocabulary" << vocabulary;
    fs_vocab.release();
    cout << "SVM addestrato!\n";

    // Test sulle immagini di test
    for (const auto& entry : fs::directory_iterator("testFiles/")) {
        Mat testImg = imread(entry.path().string());
        vector<Detection> detections = slidingWindow(testImg, svm, vocabulary);
        cout << detections.size() << endl;
        if(detections.size() == 0) {
            cout << "No detections sorry :(" << endl;
            continue;
        }
        float iouThreshold = 0.3f;  // Soglia per determinare se due box sono considerate sovrapposte
        Detection bestDetection = getDetectionWithMaxOverlap(detections, iouThreshold);

        rectangle(testImg, bestDetection.roi, Scalar(0, 255, 0), 2);

        imshow("Detections", testImg);
        waitKey(0);
    }
    return 0;
}
