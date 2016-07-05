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



using namespace std;
using namespace cv;


/*
// The main program calling all the other functions
int main (){



string src = "/home/alex/Downloads/101_ObjectCategories/butterfly/image_0003.jpg";
Mat Img = imread(src); // Read the image file
cout<<"Image read of Size " << Img.rows << "x" <<Img.cols <<endl;
Mat G,G1;
cvtColor( Img, G, CV_BGR2GRAY );
cvtColor( G, G1, CV_GRAY2BGR );

Mat Bwnn(Img, Range(50,150),Range(50,200));
Mat Bw(G1, Range(50,150),Range(50,200));
Bw.copyTo(Bwnn);

imshow("New Image",Img);
waitKey(0);



    char c;
    std::cout<<"press esc to exit! "<<std::endl;


    VideoCapture cap;
    // open the default camera, use something different from 0 otherwise;
    // Check VideoCapture documentation.
    if(!cap.open(0)) return 0;
    while(true)
    {
          Mat frame;
          cap >> frame;
          if( frame.empty() ) break; // end of video stream
          imshow("this is you, smile! :)", frame);
          c = cin.get();
          if( waitKey(1) == 27 ) break; // stop capturing by pressing ESC 
          if( c == 27) break;
    }
    // the camera will be closed automatically upon exit
    //cap.close();


    std::cout<<"exited the Camera"<<std::endl;

    return 0;

}

*/



/** Function Headers */
void detectAndDisplay( Mat frame );

/** Global variables */
String face_cascade_name = "/home/alex/Downloads/opencv/data/haarcascades/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "/home/alex/Downloads/opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
String window_name = "Capture - Face detection";

/** @function main */
int main( void )
{

/* -----for using webcam
    VideoCapture capture;
    Mat frame;

    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };

    //-- 2. Read the video stream
    capture.open( -1 );
    if ( ! capture.isOpened() ) { printf("--(!)Error opening video capture\n"); return -1; }

    while (  capture.read(frame) )
    {
        if( frame.empty() )
        {
            printf(" --(!) No captured frame -- Break!");
            break;
        }

        //-- 3. Apply the classifier to the frame
        detectAndDisplay( frame );

        int c = waitKey(10);
        if( (char)c == 27 ) { break; } // escape
    }

*/ // -----for using webcam

       // for Sliding window:
       // Parameters of your slideing window

       // Parameters of your slideing window
      string src = "/home/alex/Downloads/101_ObjectCategories/butterfly/image_0003.jpg";
      Mat Img = imread(src); // Read the image file
      cout<<"Image read of Size " << Img.rows << "x" <<Img.cols <<endl;
      Mat G,G1;
      cvtColor( Img, G, CV_BGR2GRAY );
       int windows_n_rows = 60;
       int windows_n_cols = 60;
       // Step of each window
       int StepSlide = 30;
       // IF you want to make a copy, and do not change the source image- Use clone();
       Mat DrawResultGrid= G.clone();
       
       // Feture vect for Haar elements
       vector<double> haarFeat;
       

       // Cycle row step
      for (int row = 0; row <= G.rows - windows_n_rows; row += StepSlide)
      {
        // Cycle col step
        for (int col = 0; col <= G.cols - windows_n_cols; col += StepSlide)
        {
          // There could be feature evaluator  over Windows

          // resulting window   
          Rect windows(col, row, windows_n_rows, windows_n_cols);
   
          Mat DrawResultHere = G.clone();
          
          // Draw only rectangle
          rectangle(DrawResultHere, windows, Scalar(255), 1, 8, 0);
          // Draw grid
          rectangle(DrawResultGrid, windows, Scalar(255), 1, 8, 0);

          // Show  rectangle
          namedWindow("Step 2 draw Rectangle", WINDOW_AUTOSIZE);
          imshow("Step 2 draw Rectangle", DrawResultHere);
          waitKey(100);
          //imwrite("Step2.JPG", DrawResultHere);
          
          // Show grid
   	  namedWindow("Step 3 Show Grid", WINDOW_AUTOSIZE);
          imshow("Step 3 Show Grid", DrawResultGrid);
          waitKey(100);
          //imwrite("Step3.JPG", DrawResultGrid);

          // Select windows roi
          Mat Roi = G(windows);

          //Show ROI
          namedWindow("Step 4 Draw selected Roi", WINDOW_AUTOSIZE);
          imshow("Step 4 Draw selected Roi", Roi);
          waitKey(100);
          //imwrite("Step4.JPG", Roi);
          
          // Calculate simple Haar element for face
          // get integral Image of the ROI of the Image
          Mat isumRoi;
          integral(Roi,isumRoi,CV_64F); // omitting the type param makes an int image
          
          //ViolaJohnes element for face (only two elements used)
          int aWd = 40, aHt=20, bWd = 30, bHt = 40;
          cout<< " Size of Roi is " << isumRoi.rows << " x "<< isumRoi.cols << endl;
          //for element 1 detection (eye and cheeks)
          for(int i=0;i < (isumRoi.rows-aWd) ;i+=10){
   	    for(int j=0; j < (isumRoi.cols-aHt);j+=15){
              double s1 = isumRoi.at<double>(i+aWd/2-1,j+aHt/2-1) -isumRoi.at<double>(i+aWd/2-1,j)-isumRoi.at<double>(i,j+aHt/2-1)+isumRoi.at<double>(i,j);
               
              double s2 = isumRoi.at<double>(i+aWd-1,j+aHt-1) -isumRoi.at<double>(i+aWd-1,j+aHt/2-1)-isumRoi.at<double>(i,j+aHt-1)+isumRoi.at<double>(i,j+aHt/2-1);
              cout<< " for loop ele a with i " << i << " & j Values "<< j<< endl;
              haarFeat.push_back(s1-s2);
            }
          }
          //for element 2 detection (Nose bridge)
          for(int i=0;i< (isumRoi.rows-bWd) ;i+=15){
   	    for(int j=0; j< (isumRoi.cols-bHt),;j+=10){
              double s1 = isumRoi.at<double>(i+bWd/3-1,j+bHt-1) -isumRoi.at<double>(i+bWd/3-1,j)-isumRoi.at<double>(i,j+bHt-1)+isumRoi.at<double>(i,j);
               
              double s2 = isumRoi.at<double>(i+bWd*2/3-1,j+bHt-1) -isumRoi.at<double>(i+bWd*2/3-1,j)-isumRoi.at<double>(i+bWd/3-1,j+bHt-1)+isumRoi.at<double>(i+bWd/3-1,j);
              
              double s3 = isumRoi.at<double>(i+bWd-1,j+bHt-1) -isumRoi.at<double>(i+bWd-1,j)-isumRoi.at<double>(i+bWd*2/3-1,j+bHt-1)+isumRoi.at<double>(i+bWd*2/3-1,j);
              cout<< " for loop ele b with i " << i << " & j Values "<< j<< endl;
             
              haarFeat.push_back(2.0*s2-s1-s3);
            }
          }
         ofstream outHaar("/home/alex/Downloads/101_ObjectCategories/butterfly/image_0003.jpg.txt");
          
         for(int i=0;i<haarFeat.size();i++){
           if(outHaar.is_open()){
             outHaar<< haarFeat[i] << endl;
            }     
         }
         if(outHaar.is_open()) outHaar.close();

        } // col step for loop

      } // row step for loop

    waitKey(0);
    return 0;
}



/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
    std::vector<Rect> faces;
    Mat frame_gray;

    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

    for( size_t i = 0; i < faces.size(); i++ )
    {
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

        Mat faceROI = frame_gray( faces[i] );
        std::vector<Rect> eyes;

        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );

        for( size_t j = 0; j < eyes.size(); j++ )
        {
            Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
        }
    }
    //-- Show what you got
    imshow( window_name, frame );
}


