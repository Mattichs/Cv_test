#include <opencv2/ml.hpp>
#include <filesystem>
#include <iostream>
#include <numeric>
#include "detection.h"

namespace fs = std::filesystem;
using namespace cv;
using namespace cv::ml;

using namespace std;


int main() {
    const int VOCAB_SIZE = 100;
    string datasetPath = "dataset/";
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
    string vocabPath = "bow_vocabulary.yml";
    cv::FileStorage fs_vocab(vocabPath, cv::FileStorage::WRITE);
    if (fs_vocab.isOpened()) {
        fs_vocab << "vocabulary" << vocabulary;
        fs_vocab.release();
        cout << "Vocabolario salvato in: " << vocabPath << endl;
    } else {
        cerr << "Errore: Impossibile aprire il file per salvare il vocabolario: " << vocabPath << endl;
    }

    // Istogrammi
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

    // 2. Creazione e configurazione del classificatore Random Forest
    Ptr<ml::RTrees> rf = ml::RTrees::create();

    // Imposta i parametri (questi sono valori di esempio, potresti doverli aggiustare)
    rf->setMaxDepth(1000);          // Profondità massima degli alberi
    rf->setMinSampleCount(5);     // Numero minimo di campioni per splittare un nodo
    rf->setRegressionAccuracy(0); // Impostare a 0 per la classificazione
    rf->setUseSurrogates(false);  // Di solito non necessario
    rf->setCVFolds(0);            // Numero di fold per cross-validation (0 per non usarla durante il training)
    rf->setUse1SERule(false);
    rf->setTruncatePrunedTree(false);
    // rf->setPriors(Mat());      // Puoi impostare priori se le classi sono sbilanciate
    rf->setCalculateVarImportance(true); // Calcola l'importanza delle feature (visual words)
    rf->setActiveVarCount(0);     // Numero di feature da considerare ad ogni split (0 = usa sqrt(numero totale feature))

    // Criteri di terminazione per l'addestramento degli alberi
    rf->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.01));    

    // 3. Addestramento del modello
    cout << "Addestramento Random Forest..." << endl;
    try {
        rf->train(trainData, ml::ROW_SAMPLE, labelsMat);
         cout << "Addestramento completato." << endl;
    } catch (const cv::Exception& e) {
        cerr << "Errore durante l'addestramento della Random Forest: " << e.what() << endl;
        return -1; // O gestisci l'errore
    }


    // 4. Salvare il modello addestrato
    string modelPath = "random_forest_bow_model.yml";
    cout << "Salvataggio del modello in: " << modelPath << endl;
    rf->save(modelPath);

    cout << "Modello Random Forest addestrato e salvato." << endl;


    string testPath = "test_images/";

    // testo sulle immagini 
    computeTestImages(testPath, rf, vocabulary);
            
    return 0;
}
 
