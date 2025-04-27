#include <opencv2/ml.hpp>
#include <filesystem>
#include <iostream>
#include <numeric>
#include "../include/detection.h"

namespace fs = std::filesystem;
using namespace cv;
using namespace cv::ml;

using namespace std;


int main() {
    const int VOCAB_SIZE = 100;
    string datasetPath = "../dataset/";
    vector<string> classFolders = {"mustard", "drill", "sugar"};

    vector<Mat> allDescriptors;
    vector<int> labels;
    map<string, int> classToLabel = {
        {"mustard", 0},
        {"drill", 1},
        {"sugar", 2}
    };
    
    loadImagesAndGetFeatures(datasetPath, classFolders, allDescriptors, labels, classToLabel);

    //cout << "Descriptors size: " << allDescriptors.size() <<endl;
    //cout << "Num labels: " << labels.size() <<endl;

    // Unisci tutti i descrittori
    Mat descriptorsAll;
    vconcat(allDescriptors, descriptorsAll);

    // KMeans
    cout << "KMeans clustering...\n";
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 100, 0.01);
    BOWKMeansTrainer bowTrainer(VOCAB_SIZE, criteria, 3, KMEANS_PP_CENTERS);
    Mat vocabulary = bowTrainer.cluster(descriptorsAll);
    string vocabPath = "../temp/bow_vocabulary.yml";
    cv::FileStorage fs_vocab(vocabPath, cv::FileStorage::WRITE);
    if (fs_vocab.isOpened()) {
        fs_vocab << "vocabulary" << vocabulary;
        fs_vocab.release();
        cout << "Vocabolario salvato in: " << vocabPath << endl;
    } else {
        cerr << "Errore: Impossibile aprire il file per salvare il vocabolario: " << vocabPath << endl;
    }

    // Histograms
    Mat trainData;
    for (const auto& desc : allDescriptors) {
        Mat hist = computeHistogram(desc, vocabulary);
        trainData.push_back(hist);
    }
    Mat labelsMat(labels.size(), 1, CV_32S);
    for (size_t i = 0; i < labels.size(); ++i) {
        labelsMat.at<int>(i, 0) = labels[i];
    }

    //cout << "Dimensioni TrainData: " << trainData.size() << " Tipo: " << trainData.type() << endl;
    //cout << "Dimensioni LabelsMat: " << labelsMat.size() << " Tipo: " << labelsMat.type() << endl;

    // Random Forest setup
    Ptr<ml::RTrees> rf = ml::RTrees::create();

    rf->setMaxDepth(1000);          
    rf->setMinSampleCount(5);    
    rf->setRegressionAccuracy(0); 
    rf->setUseSurrogates(false);  
    rf->setCVFolds(0);            
    rf->setUse1SERule(false);
    rf->setTruncatePrunedTree(false);
    rf->setCalculateVarImportance(true); 
    rf->setActiveVarCount(0);    
    rf->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.01));    

    // Train the model
    cout << "Addestramento Random Forest..." << endl;
    try {
        rf->train(trainData, ml::ROW_SAMPLE, labelsMat);
         cout << "Addestramento completato." << endl;
    } catch (const cv::Exception& e) {
        cerr << "Errore durante l'addestramento della Random Forest: " << e.what() << endl;
        return -1; 
    }


    // Save model
    string modelPath = "../temp/random_forest_bow_model.yml";
    cout << "Salvataggio del modello in: " << modelPath << endl;
    rf->save(modelPath);

    cout << "Modello Random Forest addestrato e salvato." << endl;


    string testPath = "../test_images/";

    cout << "Anlyzing images... Wait some seconds please :)" <<endl;
    // test on the images
    computeTestImages(testPath, rf, vocabulary);
    cout << "I'm done ;)" << endl;
    
 
    // ASSES THE PERFORMACE (probably not that good XD, we tried be kind)
    string resultsLabels = "../results/labels/";

    calcAvgIOU(resultsLabels);


    return 0;
}
 
