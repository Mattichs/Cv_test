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
const int WIN_WIDTH = 256;
const int STEP_SIZE = 16;

struct Detection {
    Rect roi;
    float prob;
    string className;
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


// Funzione per estrarre descrittori SIFT (esempio basato sul tuo codice)
cv::Mat extractSIFT(const cv::Mat& image, const cv::Mat& mask = cv::Mat()) {
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    // Nota: La maschera qui è opzionale per l'immagine di test.
    // Se la maschera era *essenziale* per definire l'oggetto nel training,
    // potresti aver bisogno di un modo per generare/fornire una maschera
    // anche per l'immagine di test, altrimenti estrai da tutta l'immagine.
    if (!mask.empty()) {
        sift->detectAndCompute(image, mask, keypoints, descriptors);
    } else {
        sift->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
    }
    return descriptors;
}

// Funzione per calcolare l'istogramma BoW (esempio basato sul tuo codice)
cv::Mat computeHistogram(const cv::Mat& descriptors, const cv::Mat& vocabulary) {
    if (descriptors.empty() || vocabulary.empty()) {
        // Restituisci un istogramma vuoto o di zeri se non ci sono descrittori
        // La dimensione deve corrispondere alla dimensione del vocabolario (VOCAB_SIZE)
        int vocabSize = vocabulary.rows; // Assumendo che VOCAB_SIZE sia la dimensione delle righe
         if (vocabSize <= 0) {
             cerr << "Errore: Dimensione del vocabolario non valida in computeHistogram." << endl;
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
    // Calcola l'istogramma BoW per i descrittori dell'immagine
    // Nota: BOWImgDescriptorExtractor richiede keypoints fittizi se si usa compute(descriptors, hist)
    // È più semplice usare l'overload che prende immagine e keypoints, ma richiede l'immagine originale.
    // Qui usiamo direttamente i descrittori, quindi dobbiamo gestire il formato.
    // In alternativa, puoi fare il matching manualmente e costruire l'istogramma.

    // Metodo Manuale (più controllo):
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

// === Sliding Window Detection ===
vector<Detection> slidingWindow(const Mat& img, Ptr<RTrees> rf, const Mat& vocab, float threshold = 0.4) {
    vector<Detection> detections;
    Ptr<SIFT> sift = SIFT::create();
    for (int y = 0; y <= img.rows - WIN_HEIGHT; y += STEP_SIZE) {
        for (int x = 0; x <= img.cols - WIN_WIDTH; x += STEP_SIZE) {
            Rect roi(x, y, WIN_WIDTH,WIN_HEIGHT);
            Mat patchColor = img(roi);
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
            int totalVotes;
            std::vector<std::string> classNames = {"mustard", "drill", "sugar"};
            std::vector<double> votesPerClass(classNames.size());
            Mat votes = predictionResults.row(1);
            //cout << votes << endl;
            float sum = 0.0f;
            for(int i = 0; i < votes.cols; i++) sum+=votes.at<int>(0,i);
            
            for (int i= 0; i < votes.cols; i++) {
                float prob = votes.at<int>(0,i)/sum;
                cout << classNames[i] << " : " << prob << endl;    
                if(prob > threshold) {
                    Detection temp;
                    temp.className = classNames[i];
                    temp.roi = roi;
                    temp.prob = prob;
                    detections.push_back(temp);
                }
            }
        
            /* imshow("patch", patchColor);      
            waitKey(0);    */       
        }
    }
    
    
    return detections;
}

bool compareByProb(const Detection &a, const Detection &b) {
    return a.prob > b.prob;
}

int main() {
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

        
        vector<Detection> detections = slidingWindow(testColor, rf, vocabulary);
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

    /*

    // SVM
    Ptr<SVM> svm = SVM::create();
    svm->setKernel(SVM::LINEAR);
    svm->setType(SVM::C_SVC);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-6));
    svm->train(trainData, ROW_SAMPLE, labels);
    svm->save("svm_model.yml");
    FileStorage fs_vocab("vocab.yml", FileStorage::WRITE);
    fs_vocab << "vocabulary" << vocabulary;
    fs_vocab.release();
    cout << "SVM addestrato!\n";

    // Test sulle immagini di test
    for (const auto& entry : fs::directory_iterator("testFiles/")) {
        Mat testImg = imread(entry.path().string());
        //Mat testImg = imread("testFiles/35_0054_001329-color.jpg");
        Mat gray, claheResult;
        cvtColor(testImg, gray, COLOR_BGR2GRAY);
        Ptr<CLAHE> clahe = createCLAHE();
        clahe->setClipLimit(4.0);
        clahe->apply(gray, claheResult);
        //Mat gray;
        //cvtColor(testImg, gray, COLOR_BGR2GRAY);
        vector<Detection> detections = slidingWindow(claheResult, svm, vocabulary);
        cout << detections.size() << endl;
        if(detections.size() == 0) {
            cout << "No detections sorry :(" << endl;
            continue;
        }
        std::sort(detections.begin(), detections.end(), compareByProb);
        // get top 10 predictions
        vector<Detection> topDetection;
        int maxDetection = 10;
        if(detections.size() < maxDetection) maxDetection = detections.size();
        for(int i = 0; i < maxDetection;i++) topDetection.push_back(detections[i]);
        
        float iouThreshold = 0.3f;  // Soglia per determinare se due box sono considerate sovrapposte
        Detection bestDetection = getDetectionWithMaxOverlap(detections, iouThreshold);
        
        rectangle(testImg, bestDetection.roi, Scalar(0, 0, 255), 2);
        imshow("Detections", testImg);
        waitKey(0);
    } */
    return 0;
}
 
