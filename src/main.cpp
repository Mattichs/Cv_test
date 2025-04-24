#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml.hpp>
#include <filesystem>
#include <iostream>
#include <numeric>
#include "detection.h"

namespace fs = std::filesystem;
using namespace cv;
using namespace cv::ml;
using namespace cv::xfeatures2d;
using namespace std;


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


bool compareByProb(const Detection &a, const Detection &b) {
    return a.prob > b.prob;
}

int main() {
    const int VOCAB_SIZE = 100;
    const int WIN_HEIGHT= 256;
    const int WIN_WIDTH = 256;
    const int STEP_SIZE = 16;
    string datasetPath = "dataset/";
    vector<string> classFolders = {"mustard", "drill", "sugar"};

    vector<Mat> allDescriptors;
    vector<int> labels;
    map<string, int> classToLabel = {
        {"mustard", 0},
        {"drill", 1},
        {"sugar", 2}
    };

    for(const auto& folder : classFolders) {
        string currentFolder = datasetPath + folder + "/";
        for(const auto& entry : fs::directory_iterator(currentFolder)) {
            string filename = entry.path().filename().string();
            if (filename.find("_color.png") == string::npos) {
                continue;
            }
            string base = filename.substr(0, filename.find("_color.png"));
            string colorPath = currentFolder + base + "_color.png";
            string maskPath = currentFolder + base + "_mask.png";

            Mat color = imread(colorPath);
            if(color.empty()) {
                cerr << "Non riesco a leggere l'immagine: " << colorPath << endl;
                continue;
            }
            Mat mask = imread(maskPath, IMREAD_GRAYSCALE);
            if(mask.empty()) {
                cerr << "Non riesco a leggere la maschera: " << maskPath << endl;
                continue;
            }
            Mat gray;
            cvtColor(color, gray, COLOR_BGR2GRAY);
            Mat claheResult;
            Ptr<CLAHE> clahe = createCLAHE();
            clahe->setClipLimit(4.0);
            clahe->apply(gray, claheResult);

            Mat desc = extractSIFT(claheResult, mask);
            if (!desc.empty()) {
                allDescriptors.push_back(desc);
                labels.push_back(classToLabel[folder]);  
            }    
        }
    }
    cout << "Num labels: " << labels.size() <<endl;
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
        // Assicurati che le etichette siano interi >= 0
        labelsMat.at<int>(i, 0) = labels[i];
    }

    // Controllo Dimensioni (opzionale ma utile per debug)
    cout << "Dimensioni TrainData: " << trainData.size() << " Tipo: " << trainData.type() << endl;
    cout << "Dimensioni LabelsMat: " << labelsMat.size() << " Tipo: " << labelsMat.type() << endl;

    // Verifica che il numero di campioni corrisponda
    if (trainData.rows != labelsMat.rows) {
        cerr << "Errore: Il numero di campioni in trainData (" << trainData.rows
             << ") non corrisponde al numero di etichette (" << labelsMat.rows << ")" << endl;
        return -1; // O gestisci l'errore come preferisci
    }
    // Verifica che trainData sia CV_32F
    if (trainData.type() != CV_32F) {
        cerr << "Avviso: trainData non è di tipo CV_32F. Tentativo di conversione." << endl;
        trainData.convertTo(trainData, CV_32F);
        if (trainData.type() != CV_32F) {
             cerr << "Errore: Impossibile convertire trainData a CV_32F." << endl;
             return -1;
        }
    }

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


    // 4. (Opzionale ma raccomandato) Salvare il modello addestrato
    string modelPath = "random_forest_bow_model.yml";
    cout << "Salvataggio del modello in: " << modelPath << endl;
    rf->save(modelPath);

    cout << "Modello Random Forest addestrato e salvato." << endl;

    std::vector<std::string> classNames = {"mustard", "drill", "sugar"}; // Stesso ordine delle etichette 0, 1, 2

    // --- Parametri e Setup ---
    for (const auto& entry : fs::directory_iterator("test_images/")) {
        // --- Caricamento e Preprocessing Immagine di Test ---
        Mat testColor = imread(entry.path().string());
        if (testColor.empty()) {
            std::cerr << "Errore: Impossibile leggere l'immagine di test: " << std::endl;
            return -1;
        }

        float treshold = 0.4;
        vector<Detection> detections = slidingWindow(testColor, rf, vocabulary, treshold, WIN_WIDTH, WIN_HEIGHT, STEP_SIZE);
        cout << detections.size() << endl;
        if(detections.size() == 0) {
            cout << "No detections sorry :(" << endl;
        }
        //std::sort(detections.begin(), detections.end(), compareByProb);
        
        // split detection to mustard, drill and sugar
        vector<Detection> mustardDetections;
        vector<Detection> drillDetections;
        vector<Detection> sugarDetections;
        for(const auto & d: detections) {
            if(d.className == "mustard") {
                mustardDetections.push_back(d);
            } else if(d.className == "drill") {
                drillDetections.push_back(d);
            } else {
                sugarDetections.push_back(d);
            }
        }
        cout << "mustard Detections:" << mustardDetections.size() << endl;
        cout << "drill Detections:" << drillDetections.size() << endl;
        cout << "sugar Detections:" << sugarDetections.size() << endl;

        int maxDetection = 10;
        float iouThreshold = 0.3f; 
        // get best for mustard
        if(!mustardDetections.empty()) {
            if(mustardDetections.size() == 1) {
                rectangle(testColor, mustardDetections[0].roi, Scalar(0, 0, 255), 2);
            } else {
                std::sort(mustardDetections.begin(), mustardDetections.end(), compareByProb);
            
                vector<Detection> topMustardDetection;
            
                if(mustardDetections.size() < maxDetection) maxDetection = mustardDetections.size();
                for(int i = 0; i < maxDetection;i++) topMustardDetection.push_back(mustardDetections[i]);
                
                // Soglia per determinare se due box sono considerate sovrapposte
                Detection bestMustardDetection = getDetectionWithMaxOverlap(topMustardDetection, iouThreshold);
                rectangle(testColor, bestMustardDetection.roi, Scalar(0, 0, 255), 2);
       
            }
        }
        
        // get best for drill
        if(!drillDetections.empty()) {
            if(drillDetections.size() == 1) {
                rectangle(testColor, drillDetections[0].roi, Scalar(255, 0, 0), 2);
            } else {
                std::sort(drillDetections.begin(), drillDetections.end(), compareByProb);
        
                vector<Detection> topDrillDetection;
                if(drillDetections.size() < maxDetection) maxDetection = drillDetections.size();
                for(int i = 0; i < maxDetection;i++) topDrillDetection.push_back(drillDetections[i]);
                
                Detection bestDrillDetection = getDetectionWithMaxOverlap(topDrillDetection, iouThreshold);
                rectangle(testColor, bestDrillDetection.roi, Scalar(255, 0, 0), 2);
            }
        }
        
        
        // get best from sugar
        if(!sugarDetections.empty()) {
            if(sugarDetections.size() == 1) {
                rectangle(testColor, sugarDetections[0].roi, Scalar(0, 255, 0), 2);
            } else {
                std::sort(sugarDetections.begin(), sugarDetections.end(), compareByProb);
                
                vector<Detection> topSugarDetection;
                if(sugarDetections.size() < maxDetection) maxDetection = sugarDetections.size();
                for(int i = 0; i < maxDetection;i++) topSugarDetection.push_back(sugarDetections[i]);
                
                Detection bestSugarDetection = getDetectionWithMaxOverlap(topSugarDetection, iouThreshold);
                rectangle(testColor, bestSugarDetection.roi, Scalar(0, 255, 0), 2);
            }
        }
        imshow("Detections", testColor);
        waitKey(0);
    }
    
    

    


    
    return 0;

    
    return 0;
}
 
