#include <iostream>
#include <string>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include <fstream>
#include <math.h>
#include <boost/foreach.hpp>
#include <numeric>
#include <sstream>



using namespace std;
using namespace cv;


/*

This is used to create the Sliding Window (60x60 px) images from all the training Images 
The Training data used here is ICCV 2009 Classification data
classes: -1 unknown, 0 sky, 1 tree/bush, 2 road/path, 3 grass, 4 water, 5 building, 6 mountain, 7 foreground object.

We are interested in only 0,1,3 classes


*/


int main (){

  string trainList = "/home/alex/Desktop/V4RL_MasterThesis/CompVis/online-random-forests-master/data/images/trainImgs.txt";
  string trainarr;
  ifstream trList(trainList);
  //int numm = 1;
  cout<<"Image read of Siz" << endl;
  if(trList.is_open())
    {
      while(getline(trList, trainarr))
      {
      //if (numm == 2) break; // for the initial test run of the code, later to be removed 


      
      Mat Img = imread("/home/alex/Desktop/V4RL_MasterThesis/CompVis/online-random-forests-master/data/images/"+trainarr); // Read the image file
      cout<<"Image read of Size " << Img.rows << "x" <<Img.cols <<endl;
      string trainarr1 = trainarr.substr(0,trainarr.size()-3);
      cout<<"Image read is " << trainarr1 << endl;
      Mat G, lab(Img.rows,Img.cols,CV_32SC1, Scalar(0));
      

      
      // Label read from file at each pixel level
      
      ifstream trLab("/home/alex/Desktop/V4RL_MasterThesis/CompVis/online-random-forests-master/data/labels/"+ trainarr1 +"regions.txt");
      string trainlab;

      if(trLab.is_open()){
        int i = 0;
      
        while(getline(trLab, trainlab)){
            istringstream ss( trainlab );
	    int j=0;
            string s;
            //ss >> s;
            

	    while (ss >> s )
	    {
               lab.at<int>(i,j) =  stoi(s);
               if (j == Img.cols-1 && i ==Img.rows-1) cout << stoi(s) << endl;
               j++;
            }
            //cout << "exit the inner loop with j = " << j << endl;
            //if (j > Img.cols) cout<< " error in the cols in the Labfile " << trainarr1 << endl; 
            i++;
            //ss.str(std::string());
            ss.str("");
            //ss.clear();
            trainlab.clear();


               
        }
        
        if (trLab.is_open()) trLab.close();
        //cout << "exit the outer loop with i = " << i << endl;
        //if (i > Img.rows) cout<< " error in the rows in the Labfile " << trainarr1 << endl; 
      }
      else cout<<" the Label file for could not be opened" << endl;
      
      
      //to write the class variable to file
      ofstream clFile("/home/alex/Desktop/V4RL_MasterThesis/CompVis/online-random-forests-master/data/imageSWindow/classVar.txt", ios::app);
      if(clFile.is_open()){

      
      cvtColor( Img, G, CV_BGR2GRAY );
      int windows_n_rows = 60;
      int windows_n_cols = 60;
      // Step of each window
      int StepSlide = 30;
      // IF you want to make a copy, and do not change the source image- Use clone();
      Mat DrawResultGrid= G.clone();
      //imshow("Gray", G);
      //waitKey(0);
      // Feture vect for Haar elements
      cout << G.rows - windows_n_rows << " " << G.cols - windows_n_cols << endl;
      int winNum = 1; // windows number per image
       // Cycle row step
      for (int row = 0; row <= G.rows - windows_n_rows; row += StepSlide)
      {
        // Cycle col step
        cout << "entered the first loop" << endl;
        for (int col = 0; col <= G.cols - windows_n_cols; col += StepSlide)
        {
          cout << "entered the second loop" << endl;
          // There could be feature evaluator  over Windows
/*
          // resulting window   
          Rect windows(col, row, windows_n_rows, windows_n_cols);
   
          Mat DrawResultHere = G.clone();
          
          // Draw only rectangle
          rectangle(DrawResultHere, windows, Scalar(255), 1, 8, 0);
          // Draw grid
          rectangle(DrawResultGrid, windows, Scalar(255), 1, 8, 0);

          // Show  rectangle
          //namedWindow("Step 2 draw Rectangle", WINDOW_AUTOSIZE);
          //imshow("Step 2 draw Rectangle", DrawResultHere);
          //waitKey(100);
          //imwrite("Step2.JPG", DrawResultHere);
          
          // Show grid
   	  //namedWindow("Step 3 Show Grid", WINDOW_AUTOSIZE);
          //imshow("Step 3 Show Grid", DrawResultGrid);
          //waitKey(100);
          //imwrite("Step3.JPG", DrawResultGrid);

          // Select windows roi
          Mat Roi = G(windows);
          
          //Show ROI
          //namedWindow("Step 4 Draw selected Roi", WINDOW_AUTOSIZE);
          //imshow("Step 4 Draw selected Roi", Roi);
          //waitKey(100);
          */
          //string outfile = "/home/alex/Desktop/V4RL_MasterThesis/CompVis/online-random-forests-master/data/imageSWindow/"+trainarr1+to_string(winNum) + ".jpg";  // "."+ to_string(lab.at<int>(row+30,col+30)) + 
          //imwrite(outfile, Roi);
          clFile << trainarr1+to_string(winNum) << " " << lab.at<int>(row+30,col+30) << endl;
         
          winNum++; // increment the window number
        } // inner for loop
      }   // outer for loop
     if (clFile.is_open()) clFile.close();
     }// if loop of Class Var file
     else cout << "could not open the class file to write" << endl;


      G.release();
      lab.release();
      Img.release();
      //numm++;
      //trainarr1.clear();
      //trainlab.clear();
      trainarr.clear();
      trainList.clear();
      } // while of trList
      if (trList.is_open()) trList.close();
  }// if loop
  else {
    cout << "could not open trainList file" << endl;
  }
  
  
  

return 0;

}

