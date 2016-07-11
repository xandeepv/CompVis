#define GMM_USES_BLAS

#include <cstdlib>
#include <iostream>
#include <time.h>
#include <string>
#include <string.h>
#include <libconfig.h++>
#include <sstream>
#include "data.h"
#include "onlinetree.h"
#include "onlinerf.h"
#include <bzlib.h>
#include <fstream>
#include <stdlib.h>
#include <algorithm>
#include <functional>
#include <cctype>
#include <locale>


using namespace std;
using namespace libconfig;


/*
 * Compile using the below code:
 * g++ -std=gnu++11 Online-Forest_xan.cpp `pkg-config opencv --cflags --libs` -lconfig++ -latlas -llapack -lblas -lbz2
 */

typedef enum {
    ORT, ORF
} CLASSIFIER_TYPE;


// trim from start
static inline std::string &ltrim(std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
        return s;
}

// trim from end
static inline std::string &rtrim(std::string &s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
        return s;
}

//! Returns the time (ms) elapsed between two calls to this function
double timeIt(int reset) {
    static time_t startTime, endTime;
    static int timerWorking = 0;

    if (reset) {
        startTime = time(NULL);
        timerWorking = 1;
        return -1;
    } else {
        if (timerWorking) {
            endTime = time(NULL);
            timerWorking = 0;
            return (double) (endTime - startTime);
        } else {
            startTime = time(NULL);
            timerWorking = 1;
            return -1;
        }
    }
}

int main(int argc, char *argv[]) {
    // Parsing command line
    string confFileName="/home/alex/Desktop/V4RL_MasterThesis/CompVis/online-random-forests-master/conf/orf.conf";
    int classifier = -1, doTraining = false, doTesting = false, doT2 = false, inputCounter = 1;

    classifier = ORF; //ORT for tree
    doTraining = true;
    doTesting = true;


    /*
    while (inputCounter < argc) {
        if (!strcmp(argv[inputCounter], "-h") || !strcmp(argv[inputCounter], "--help")) {
            help();
            return EXIT_SUCCESS;
        } else if (!strcmp(argv[inputCounter], "-c")) {
            confFileName = argv[++inputCounter];
        } else if (!strcmp(argv[inputCounter], "--ort")) {
            classifier = ORT;
        } else if (!strcmp(argv[inputCounter], "--orf")) {
            classifier = ORF;
        } else if (!strcmp(argv[inputCounter], "--train")) {
            doTraining = true;
        } else if (!strcmp(argv[inputCounter], "--test")) {
            doTesting = true;
        } else if (!strcmp(argv[inputCounter], "--t2")) {
            doT2 = true;
        } else {
            cout << "\tUnknown input argument: " << argv[inputCounter];
            cout << ", please try --help for more information." << endl;
            exit(EXIT_FAILURE);
        }

        inputCounter++;
    }
    */
    cout << "OnlineMCBoost Classification Package:" << endl;

    if (!doTraining && !doTesting && !doT2) {
        cout << "\tNothing to do, no training, no testing !!!" << endl;
        exit(EXIT_FAILURE);
    }

    if (doT2) {
        doTraining = false;
        doTesting = false;
    }

    // Load the hyperparameters
    Hyperparameters hp(confFileName);

    // Creating the train data
    DataSet dataset_tr, dataset_ts;



    //dataset_tr.loadLIBSVM(hp.trainData);
    //if (doT2 || doTesting) {
    //    dataset_ts.loadLIBSVM(hp.testData);
    //}


    ifstream fileList("/home/alex/Downloads/caltech/splits/trainfiles.csv");
    string trainarr;
    ifstream labelList("/home/alex/Downloads/caltech/splits/trainLabelN1.csv");

    // params for Train Dataset
	  dataset_tr.m_numSamples = 1;
	  dataset_tr.m_numFeatures = 639;
	  dataset_tr.m_numClasses = 102;


    if(fileList.is_open() && labelList.is_open())
    {
      while(getline(fileList, trainarr))
    	  {
		   // Reading the header






		  // for each sample
		  //for (int i = 0; i < m_numSamples; i++) {
			  wsvector<double> x(dataset_tr.m_numFeatures);
			  Sample sample;
			  resize(sample.x, dataset_tr.m_numFeatures);
			  labelList >> sample.y; // read label
			  //sample.y=1; // read label
			  sample.w = 1.0; // set weight

		    string fname = "/home/alex/Downloads/caltech101_features_PHOG/phog/A360_K40/Level2/"+trainarr+".bz2";
		    cout << fname << endl;
		    FILE *f1;
		    int bzError;
		    BZFILE *bzf;
		    char buf[8192];
		    f1 = fopen(fname.c_str(), "r");
		    bzf = BZ2_bzReadOpen(&bzError, f1, 0, 0, NULL, 0);
		    if (bzError != BZ_OK) {
		      fprintf(stderr, "E: BZ2_bzReadOpen: %d\n", bzError);
		      return -1;
		    }
		    while (bzError == BZ_OK) {
		      int nread = BZ2_bzRead(&bzError, bzf, buf, sizeof buf);
		      if (bzError == BZ_OK || bzError == BZ_STREAM_END) {

		    	  istringstream ss( buf );
		    	        int i=0;
		    	        while (ss && i < 639)
		    	        {
		    	          string s;
		    	          if (!getline( ss, s, ',' )) break;

		    	          x[i] = atof(rtrim(s).c_str());
		    	          //cout << " " << i << ":" << s ;
		    	          i++;
		    	          //cout << i << endl;
		    	          if (i >= 640) cout << s << endl;
		    	        }
		    	        copy(x, sample.x);
		    	        dataset_tr.m_samples.push_back(sample); // push sample into database

		      }
		    }

		    if (bzError != BZ_STREAM_END) {
		      fprintf(stderr, "E: bzip error after read: %d\n", bzError);
		      return -1;
		    }

		    BZ2_bzReadClose(&bzError, bzf);
		    fclose(f1);

           /*
		   if (dataset_tr.m_numSamples != (int) dataset_tr.m_samples.size()) {
			  cout << "Could not load " << dataset_tr.m_numSamples << " samples from files";
			  cout << ". There were only " << dataset_tr.m_samples.size() << " samples!" << endl;
			  exit(EXIT_FAILURE);

		  }
		   */

		  // Find the data range
		  dataset_tr.findFeatRange();
		  dataset_tr.m_numSamples =  dataset_tr.m_numSamples + 1;

		  }// while loop end
    } // Training data file list ifstream is open
    fileList.close();
    labelList.close();
    dataset_tr.m_numSamples =  dataset_tr.m_numSamples - 1;
    cout << "Loaded training " << dataset_tr.m_numSamples << " samples with " << dataset_tr.m_numFeatures;
    cout << " features and " << dataset_tr.m_numClasses << " classes." << endl;





    // Do training of the Model
    OnlineRF model(hp, dataset_tr.m_numClasses, dataset_tr.m_numFeatures, dataset_tr.m_minFeatRange, dataset_tr.m_maxFeatRange);
    //OnlineTree model(hp, dataset_tr.m_numClasses, dataset_tr.m_numFeatures, dataset_tr.m_minFeatRange, dataset_tr.m_maxFeatRange);


    if (doTraining) {
    	timeIt(1);
    	model.train(dataset_tr);
    	cout << "Training time: " << timeIt(0) << endl;
    }
    

    //save the trained OnlineRF to file
    ofstream clfile("/home/alex/Desktop/V4RL_MasterThesis/CompVis/online-random-forests-master/Trained.OnlineRF.bin",ios::in|ios::out|ios::binary);
    clfile.write((char*)&model,sizeof(OnlineRF));
    clfile.close();
    //OnlineRF model1(hp, dataset_tr.m_numClasses, dataset_tr.m_numFeatures, dataset_tr.m_minFeatRange, dataset_tr.m_maxFeatRange);
    

    // For Test Data
    ifstream fileList1("/home/alex/Downloads/caltech/splits/testfiles.csv");
    string testarr;
    ifstream labelList1("/home/alex/Downloads/caltech/splits/testLabelN1.csv");

    // params for Test Dataset
	  dataset_ts.m_numSamples = 1;
	  dataset_ts.m_numFeatures = 639;
	  dataset_ts.m_numClasses = 102;


    if(fileList1.is_open() && labelList1.is_open())
    {
       while(getline(fileList1, testarr))
    	  {
		   // Reading the header




		  // Reading the data
		  //dataset_ts.m_samples.clear();

		  // for each sample
		  //for (int i = 0; i < m_numSamples; i++) {
			  wsvector<double> x(dataset_ts.m_numFeatures);
			  Sample sample;
			  //swap(sample.x, x);
			  resize(sample.x, dataset_ts.m_numFeatures);
			  labelList1 >> sample.y; // read label
			  sample.w = 1.0; // set weight

		  string fname = "/home/alex/Downloads/caltech101_features_PHOG/phog/A360_K40/Level2/"+testarr+".bz2";
		    cout << fname << endl;
		    FILE *f1;
		    int bzError;
		    BZFILE *bzf;
		    char buf[8192];
		    f1 = fopen(fname.c_str(), "r");
		    bzf = BZ2_bzReadOpen(&bzError, f1, 0, 0, NULL, 0);
		    if (bzError != BZ_OK) {
		      fprintf(stderr, "E: BZ2_bzReadOpen: %d\n", bzError);
		      return -1;
		    }
		    while (bzError == BZ_OK) {
		      int nread = BZ2_bzRead(&bzError, bzf, buf, sizeof buf);
		      if (bzError == BZ_OK || bzError == BZ_STREAM_END) {

		    	  istringstream ss( buf );
		    	        int i=0;
		    	        while (ss && i < 639)
		    	        {
		    	          string s;
		    	          if (!getline( ss, s, ',' )) break;
		    	          //s.erase(std::remove(s.begin(), s.end(), '\n'), s.end());
		    	          x[i] = atof(rtrim(s).c_str());
		    	          //cout << " " << i << ":" << s ;
		    	          i++;
		    	        }
		    	        copy(x, sample.x);
		    	        dataset_ts.m_samples.push_back(sample); // push sample into database

		      }
		    }
		    if (bzError != BZ_STREAM_END) {
		      fprintf(stderr, "E: bzip error after read: %d\n", bzError);
		      return -1;
		    }

		    BZ2_bzReadClose(&bzError, bzf);
		    fclose(f1);

          /*
		  if (dataset_ts.m_numSamples != (int) dataset_ts.m_samples.size()) {
			  cout << "Could not load " << dataset_ts.m_numSamples << " samples from files";
			  cout << ". There were only " << dataset_ts.m_samples.size() << " samples!" << endl;
			  exit(EXIT_FAILURE);
		  }
          */
		  // Find the data range
		  dataset_ts.findFeatRange();
		  dataset_ts.m_numSamples =  dataset_ts.m_numSamples + 1;

		}// while loop end

    } // Training data file list ifstream is open
    fileList1.close();
    labelList.close();
	dataset_ts.m_numSamples =  dataset_ts.m_numSamples - 1;
	  cout << "Loaded Test " << dataset_ts.m_numSamples << " samples with " << dataset_ts.m_numFeatures;
	  cout << " features and " << dataset_ts.m_numClasses << " classes." << endl;


	    cout << "-------------------------------------" << endl;
	  	cout << "-------Test Trial 1:-----------------" << endl;
	  	cout << "-------------------------------------" << endl;



	if (doTesting) {
		timeIt(1);
		vector<Result> resu = model.test(dataset_ts);
		cout << "Test time: " << timeIt(0) << endl;
		//for(int i =0; i<resu.size(); i++){
		for(int i =2000; i<2005; i++){
		  cout <<"For first pic: Predicted class=" << resu[i].prediction << endl;
		  cout << "The confidence values are: (of total tree " << resu[i].confidence.size() << endl;
		  for(int j=0;j< resu[i].confidence.size(); j++){
			  cout << resu[i].confidence[j] << " " ;
		  }
		  cout << endl;
		}
	}
	    cout << "-------------------------------------" << endl;
		cout << "-------Test Trial 2:-----------------" << endl;
		cout << "-------------------------------------" << endl;

	if (doTesting) {
		timeIt(1);
		vector<Result> resu = model.test(dataset_ts);
		cout << "Test time: " << timeIt(0) << endl;
		//for(int i =0; i<resu.size(); i++){
		for(int i =300; i<305; i++){
		  cout <<"For first pic: Predicted class=" << resu[i].prediction << endl;
		  cout << "The confidence values are: (of total tree " << resu[i].confidence.size() << endl;
		  for(int j=0;j< resu[i].confidence.size(); j++){
			  cout << resu[i].confidence[j] << " " ;
		  }
		  cout << endl;
		}
	}
	cout << "-------------------------------------" << endl;
	cout << "-------Test Trial 3:-----------------" << endl;
	cout << "-------------------------------------" << endl;
	if (doTesting) {
		timeIt(1);
		vector<Result> resu = model.test(dataset_ts);
		cout << "Test time: " << timeIt(0) << endl;
		//for(int i =0; i<resu.size(); i++){
		for(int i =1000; i<1005; i++){
		  cout <<"For first pic: Predicted class=" << resu[i].prediction << endl;
		  cout << "The confidence values are: (of total tree " << resu[i].confidence.size() << endl;
		  for(int j=0;j< resu[i].confidence.size(); j++){
			  cout << resu[i].confidence[j] << " " ;
		  }
		  cout << endl;
		}
	}


    return EXIT_SUCCESS;
}
