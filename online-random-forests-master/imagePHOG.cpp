#include <iostream>
#include <string>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include <fstream>
#include <math.h>
#include <boost/foreach.hpp>
#include <numeric>
#include <iomanip>



using namespace std;
using namespace cv;

Mat bh; //= create(Img.rows,Img.cols,CV_32FC1);
Mat bv; // = create(Img.rows,Img.cols,CV_32FC1);
//Mat Gr, E, A;

//Mat bh = create(X,Y, CV_16UC1,1);
//Mat bv = create(X,Y, CV_32FC1,0.0);


// For 4 nearest Neighbour connected in Edge Matrix space
//findConnectedLabels(L.at<int>(i,j),startLab,Bwn,i,j,m,n);
Mat findConnectedLabels(Mat L,int startLabel,Mat bwcur,int i,int j ,int m,int n){
  int a,aa,b,bb,c,cc,d,dd;

  int rnum = bwcur.rows;
  int cnum = bwcur.cols;

  if (startLabel > 1000) return L; 

/*
  // no connected values calculated at border 
  if (i==rnum) {
    a=0; aa =0;
  }
  else {
    a = bwcur.at<uchar>(i+1, j);// next row
    aa = L.at<uchar>(i+1, j);  // next row
  }
  if (i == 0) {
     b=0; bb=0;
  }
  else {
     b = bwcur.at<uchar>(i-1, j);    // previous row
     bb = L.at<uchar>(i-1, j);    // previous row
  }
  if (j==cnum) {
    c=0; cc=0;
  }
  else {
    c = bwcur.at<uchar>(i, j+1);  // next col
    cc = L.at<uchar>(i, j+1);  // next col
  }
  if (j ==0) {
    d=0;dd=0;
  }
  else {
    d = bwcur.at<uchar>(i, j-1);   // prev column
    dd = L.at<uchar>(i, j-1);   // prev column
  }
*/
    a = bwcur.at<int>(i+1, j);// next row
    aa = L.at<unsigned int>(i+1, j);  // next row
    b = bwcur.at<int>(i-1, j);    // previous row
    bb = L.at<unsigned int>(i-1, j);    // previous row
    c = bwcur.at<int>(i, j+1);  // next col
    cc = L.at<unsigned int>(i, j+1);  // next col
    d = bwcur.at<int>(i, j-1);   // prev column
    dd = L.at<unsigned int>(i, j-1);   // prev column


        if((a==1)&&(aa==0)){
            L.at<unsigned int>(i+1, j)=startLabel;
            L=findConnectedLabels(L,startLabel,bwcur,i+1,j,m,n);
        }
        
        if((b==1)&&(bb==0)){
            L.at<unsigned int>(i-1, j)=startLabel;
            L=findConnectedLabels(L,startLabel,bwcur,i-1,j,m,n);
        }
        
        if((c==1)&&(cc==0)) {
            L.at<unsigned int>(i, j+1)=startLabel;
            L=findConnectedLabels(L,startLabel,bwcur,i,j+1,m,n);
        }
        
        if((d==1)&&(dd==0)) {
            L.at<unsigned int>(i, j-1)=startLabel;
            L=findConnectedLabels(L,startLabel,bwcur,i,j-1,m,n);
        }
   return L;
}








//The function to create C++ equivalent of Matlab BWLABELS func
Mat detectBw(Mat Bw1){
   int m = Bw1.rows;
   int n = Bw1.cols; 
   cout<<"Edge Binary2 size is "<< m << "x" << n << endl;
  
   //Mat Res;
   Mat Bwn = Mat(m+2,n+2, CV_8UC1, Scalar(0));

   cout<<"Edge Binary3 size is "<< Bwn.rows << "x" << Bwn.cols << endl;
   

   Mat Bwnn = Mat(Bwn, Range(1,m+1),Range(1,n+1));
   cout<<"Bwnn Created: " << Bwnn.rows << "x" << Bwnn.cols <<endl;
   
   Bw1.copyTo(Bwnn);
   cout<<"Bw1 copied to Bwnn" << endl;
   
   cout<<"size of F mat "<< m << "x" << n << endl;
   Mat L(m+2,n+2, CV_16UC1, Scalar(0));
   cout<<"L Created" << endl;
   




   unsigned int startLab =1;
   cout<< " detectBw: Size of L mat " << L.rows << "x"<< L.cols << endl;
   //cout<< " detectBw: Size of B mat " << L.rows << "x"<< L.cols << endl;
   for(int i =1; i< m+1; i++){
     for (int j=1; j<n+1; j++){
       int curdata = Bwn.at<uchar>(i,j);
       int lc = L.at<unsigned int>(i,j);
       //cout<< " for loop detectBw with i " << i << " & j Values "<< j<< endl;
       if((curdata==1)&&(lc==0)){
          L.at<unsigned int>(i,j) = startLab;
          L=findConnectedLabels(L,startLab,Bwn,i,j,m,n);
          ++startLab;
       }
     }
     if(startLab > 1000) break;
   }
   cout<< " Completed BWLABELS"<< endl;
   Mat Resu1(L,Range(1,m+1),Range(1,n+1));
   Resu1.at<unsigned int>(0,0) = startLab;	
   cout<< " sent back BWLABELS Results"<< endl;
   //Bwn.release();
   //L.release();
   //Bwnn.release();
   //Bw1.release();
   cout<<"BWLABEL result size "<< Resu1.rows << "x" << Resu1.cols << endl;
   

/*  ----------------Print Resu1 Values

   string pfile1 = "/home/alex/Downloads/101_ObjectCategories/accordion/image_0001.jpg.p.txt";
   ofstream datfile1(pfile1);
   if (datfile1.is_open()) {
     cout << " Printing L Mat values to file:" << endl;
     for(int i = 0; i< Resu1.rows; i++){
       for(int j = 0; j< Resu1.cols; j++){  
          datfile1 << Resu1.at<int>(i,j) << " " ;
       }
       datfile1 << endl;
     }
   }
   else {
   cout<<" fail to open Data file to print PHOG"<< endl;
   }
   if (datfile1.is_open()) datfile1.close();





*/

   return Resu1;
}







//vgg_binMatrix(A,E,Gr,angle,bin)

void vgg_binMatrix(Mat A1, Mat E1,Mat Gr1,Mat F, int angle,int bin){
//void vgg_binMatrix(int angle,int bin){
  //imshow("Edge Mat",E);
  cout<<"Edge size is "<< E1.rows << "x" << E1.cols << endl;

  //imshow("Edge Mat",E);
  cout<<"Angle size is "<< A1.rows << "x" << A1.cols << endl;
  cout<<"Gradient size is "<< Gr1.rows << "x" << Gr1.cols << endl;
/*
% VGG_BINMATRIX Computes a Matrix (bh) with the same size of the image where
% (i,j) position contains the histogram value for the pixel at position (i,j)
% and another matrix (bv) where the position (i,j) contains the gradient
% value for the pixel at position (i,j)
%                
%IN:
%	A - Matrix containing the angle values
%	E - Edge Image
%   G - Matrix containing the gradient values
%	angle - 180 or 360%   
%   bin - Number of bins on the histogram 
%	angle - 180 or 360
%OUT:
%	bh - matrix with the histogram values
%   bv - matrix with the graident values (only for the pixels belonging to
%   and edge)
*/
  //Mat dstIm;





 // threshold( E1, dstIm, 5, 255,THRESH_BINARY );
//imshow("Edge Mat",E1);
cout<<"Edge size is "<< E1.rows << "x" << E1.cols << endl;
//imshow("Edge Binary",dstIm);
//cout<<"Edge Binary size is "<< dstIm.rows << "x" << dstIm.cols << endl;

Mat L = detectBw(F);  
int X = E1.rows;
int Y = E1.cols;


   string pfile4 = "/home/alex/Downloads/101_ObjectCategories/accordion/image_0001.jpg.L.txt";
   ofstream datfile4(pfile4);
   if (datfile4.is_open()) {
     cout << " Printing L Mat values to file:" << endl;
     for(int i = 0; i< L.rows; i++){
       for(int j = 0; j< L.cols; j++){  
            datfile4 << L.at<int>(i,j) << " " ;
       }
       datfile4 << endl;
     }
   }
   else {
   cout<<" fail to open Data file to print L matrix"<< endl;
   }
   if (datfile4.is_open()) datfile4.close();


double nAngle = (double)(angle/bin);
cout << "nAngle" << nAngle << endl;
for(int cntNum =1; cntNum < L.at<unsigned int>(0,0);cntNum++){
for(int i=0; i <X; i++) { 
    for(int j=0; j < Y;j++) {
      if(L.at<unsigned int>(i,j) == cntNum){
        int b = ceil(A1.at<double>(i,j)/nAngle);
        if (b==0) bin= 1;
        //cout<< " for loop binMatrix with i " << i << " & j Values "<< j<< endl;
        if (Gr1.at<double>(i,j)>0.0){
            bh.at<unsigned int>(i,j) = b;
            bv.at<double>(i,j) = Gr1.at<double>(i,j);                
        }
      }
    }
}
}
//E1.release();
//A1.release();
//Gr1.release();
cout<< " Completed Binmatrix Exec"<< endl;


}





//vgg_phogDescriptor(bh_roi,bv_roi,Level,bin)

vector<double> vgg_phogDescriptor(Mat bh1, Mat bv1,int Level,int bin){
/*
function p = vgg_phogDescriptor(bh,bv,Level,bin)
% VGG_PHOGDESCRIPTOR Computes Pyramid Histogram of Oriented Gradient over a ROI.
%               
%IN:
%	bh1 - matrix of bin histogram values of ROI
%	bv1 - matrix of gradient values of ROI
%       Level - number of pyramid levels
%       bin - number of bins
%
%OUT:
%	p - pyramid histogram of oriented gradients (phog descriptor)
*/
vector<double> p;
// level 0
for(int b=1; b<=bin; b++){
    double sum1 = 0.0;
    for(int i = 0; i < bv1.rows; i++){
      for(int j = 0; j < bv1.cols; j++){ 
           if (bh1.at<unsigned int>(i,j) == b) sum1 = sum1 + bv1.at<double>(i,j);
      }
    }
    p.push_back(sum1);
    cout << fixed << setprecision(6) << sum1;
}
//if(Level >= 1){ 
int cella = 1;
for(int l=1; l <=Level; l++){
    int x = bh1.rows/(2^l);
    int y = bh1.cols/(2^l);
    int xx=0;
    int yy=0;
    while (xx+x<bh1.rows){
        while (yy +y < bh1.cols){
            
            Mat bh_cella(bh1,Range(xx+1,xx+x),Range(yy+1,yy+y));
            Mat bv_cella(bv1,Range(xx+1,xx+x),Range(yy+1,yy+y));
            
            for(int b=1; b<= bin; b++){

               double sum1 = 0.0;
               for(int i = 0; i < bh_cella.rows; i++){
                 for(int j = 0; j < bh_cella.cols; j++){ 
                    if (bh_cella.at<int>(i,j) == b) sum1 = sum1 + bv_cella.at<double>(i,j);
                 }
               }
               p.push_back(sum1);
            } 
            yy = yy+y;
        }        
        cella++;
        yy = 0;
        xx = xx+x;
    }
}
//}
double p_sum = std::accumulate(p.begin(), p.end(), 0.0);
cout << "The sum of vector at levels: " << p_sum << endl;
if (p_sum != 0.0 ){
    std::transform(p.begin(), p.end(), p.begin(), std::bind2nd(std::multiplies<double>(), 1.0/p_sum));
}
return p;
}







//vgg_phog
//vector<double> vgg_phog(string src, const Mat Img, int bin, int angle, int Level, int roi[4]){

void vgg_phog(string src, Mat Img, int bin, int angle, int Level, int roi[4]){
/*
% VGG_PHOG Computes Pyramid Histogram of Oriented Gradient over a ROI.
%               
% [BH, BV] = VGG_PHOG(I,BIN,ANGLE,L,ROI) computes phog descriptor over a ROI.
% 
% Given and image I, phog computes the Pyramid Histogram of Oriented Gradients
% over L pyramid levels and over a Region Of Interest

%IN:
%       src - the path of the image
%	I - Images of size MxN (Color or Gray)
%	bin - Number of bins on the histogram 
%	angle - 180 or 360
%       Level - number of pyramid levels
%       roi - Region Of Interest (ytop,ybottom,xleft,xright)
%
%OUT:
%	p - pyramid histogram of oriented gradients
*/
Mat G, x2y2_sum, Gr, E, A, G_EqHist, dsThresh, F;


Mat detected_edges,detected_edges1, x_derivative, y_derivative, x2_derivative, y2_derivative,xy_derivative, YX ;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 200;
int ratio = 3;
int kernel_size = 3;
//char* window_name = "Edge Map";


if (Img.channels() == 3) {
cvtColor( Img, G, CV_BGR2GRAY ); //cout<< "Image to Gray" << endl;
}
else G = Img;
//cout<< "Gray size:" << G.rows << "x" <<G.cols <<endl;


// no roi given, use the entire image
if ( roi[0] == 0 && roi[1] == 0 && roi[2] ==0 && roi[3] == 0) {
roi[1] = Img.rows;
roi[3]=Img.cols;
}

    imshow("Gray Map", G);
    waitKey(0);


if (sum(G)[0]>100.0) {

    /// Reduce noise with a kernel 3x3
    blur( G, detected_edges, Size(3,3) );

    /// Canny detector
    //Canny( detected_edges, detected_edges1, lowThreshold, lowThreshold*ratio, kernel_size );
    Canny( detected_edges, detected_edges1, -10^6, 180, kernel_size );
    /// Using Canny's output as a mask, we display our result
    E = Mat(detected_edges1.rows, detected_edges1.cols, CV_16UC1, Scalar(0));
    cout<< "Detect Edge size:" << detected_edges1.rows << "x" <<detected_edges1.cols <<endl;
    G.copyTo( E, detected_edges1);
    F = Mat(detected_edges1.rows, detected_edges1.cols, CV_8UC1, Scalar(0));
    G.copyTo(F,detected_edges1);
    cout<< "G mat:" << G.rows << "x" <<G.cols <<endl;
    cout<< "E mat:" << E.rows << "x" <<E.cols <<endl;

    imshow("Edge Map", E);
 
    equalizeHist(E, G_EqHist); 
    imshow("Edge Map - Histogram", G_EqHist);
    imshow("Edge Map - Detected ", detected_edges);
    waitKey(0);
   threshold( G, dsThresh, 100, 255,THRESH_BINARY );
   
   //definition of F vector for edges
   for(int i = 0; i< detected_edges1.rows; i++){
       for(int j = 0; j< detected_edges1.cols; j++){  
          if (F.at<uchar>(i,j) > 0) {
            F.at<uchar>(i,j) = 1;
          }
          else {
            F.at<uchar>(i,j) = 0;
          }
       }
   }

    imshow("Edge Map- F", F);
    imshow("Edge Map - Binary ", dsThresh);
    waitKey(0);



    //Step one
    //to calculate x and y derivative of image we use Sobel function
    //Sobel( srcimage, dstimage, depthofimage -1 means same as input, xorder 1,yorder 0,kernelsize 3, BORDER_DEFAULT);
    Sobel(G, x_derivative, CV_32FC1 , 1, 0, 3, BORDER_DEFAULT);
    Sobel(G, y_derivative, CV_32FC1 , 0, 1, 3, BORDER_DEFAULT);
    cout<< "D(x) size:" << x_derivative.rows << "x" <<x_derivative.cols <<endl;
    cout<< "D(y) size:" << y_derivative.rows << "x" <<y_derivative.cols <<endl;
    //Step Two calculate other three images in M
    pow(x_derivative,2.0,x2_derivative);
    pow(y_derivative,2.0,y2_derivative);
    cout<< "D2(y) size:" << x2_derivative.rows << "x" <<x2_derivative.cols <<endl;
    cout<< "D2(y) size:" << y2_derivative.rows << "x" <<y2_derivative.cols <<endl;
    //multiply(x_derivative,y_derivative,xy_derivative);
    x2y2_sum = x2_derivative + y2_derivative; 
    sqrt(x2y2_sum, Gr);
    cout<< "x2y2_sum size:" << x2y2_sum.rows << "x" <<x2y2_sum.cols <<endl;
    x_derivative.setTo(0.00001, x_derivative == 0);
        
            
    //YX = GradientY./GradientX;
    divide(y_derivative, x_derivative, YX, 1, -1);
    cout<< "YX size:" << YX.rows << "x" << YX.cols <<endl;

    A= Mat(YX.rows,YX.cols,CV_32FC1, Scalar(0.0));

    for (int i =0;i< YX.rows;i++){
      for (int j = 0;j < YX.cols ; j++) {
        if (angle == 180) A.at<double>(i,j) = ((atan(YX.at<double>(i,j))+(M_PI/2.0))*180.0)/M_PI; 
        if (angle == 360) A.at<double>(i,j) = ((atan2(y_derivative.at<double>(i,j),x_derivative.at<double>(i,j))+M_PI)*180.0)/M_PI;
            //cout<< "Angle Mat compute i: " << i << " & j: " << j <<endl;
      }
    }
    cout<< "E mat:" << E.rows << "x" <<E.cols <<endl;

   string pfile3 = "/home/alex/Downloads/101_ObjectCategories/accordion/image_0001.jpg.p.txt";
   ofstream datfile3(pfile3);
   if (datfile3.is_open()) {
     cout << " Printing A Mat values to file:" << endl;
     for(int i = 0; i< A.rows; i++){
       for(int j = 0; j< A.cols; j++){  
            datfile3 << A.at<double>(i,j) << " " ;
       }
       datfile3 << endl;
     }
   }
   else {
   cout<<" fail to open Data file to print A matrix"<< endl;
   }
   if (datfile3.is_open()) datfile3.close();






    //const Mat Ed = E, Gr1=Gr,An=A;
    detected_edges.release();
    x_derivative.release();
    //y_derivative.release();
    //x2_derivative.release();
    //y2_derivative.release();
    //xy_derivative.release(); 
    //YX.release();
    //x2y2_sum.release();
    cout<< "Calling binMatrix" <<endl;
    vgg_binMatrix(A,E,Gr,F, angle,bin);
    //vgg_binMatrix(A,G_EqHist,Gr,angle,bin);
 }

    cout<< "received binMatrix results" <<endl;

//else {
//    Mat bh = create(Img.rows,Img.cols,CV_32FC1);
//    Mat bv = create(Img.rows,Img.cols,CV_32FC1);
//}
 cout<< "ROI is: ["<<roi[0]<<"," <<roi[1] <<"]x["<<roi[2] << "," << roi[3] << "]" <<endl;
Mat bh_roi(bh, Range(roi[0],roi[1]),Range(roi[2],roi[3]));
Mat bv_roi(bv, Range(roi[0],roi[1]),Range(roi[2],roi[3]));

    cout<< "bh_roi: " << bh_roi.rows << "x" <<bh_roi.cols <<endl;
    cout<< "bv_roi: " << bv_roi.rows << "x" <<bv_roi.cols <<endl;


string pfile1 = src+"bv_vec.txt";
cout<< " Loc of BV Vect File: " << pfile1 << endl;
cout<< " Size of BV Vect: " << bv.rows<< "x" <<bv.cols << endl;
ofstream datfile1(pfile1);
if (datfile1.is_open()) {
   cout << " Printing BV values to file:" << endl;
   datfile1 << fixed << setprecision(6);
  for(int i = 0; i < bv.rows; i++){
     for(int j = 0; j < bv.cols; j++){
       datfile1 << bv.at<double>(i,j) << " "; 
   //for (auto i = p.begin(); i != p.end(); i++){
    //datfile << *i << endl
     }
     datfile1 << endl;
   }
}
else {
cout<<" fail to open Data file to print PHOG"<< endl;
}
if (datfile1.is_open()) datfile1.close();
cout<<" Completed writin BV to file, CLeanup" << endl;




string pfile2 = src+"bh_Vec.txt";
cout<< " Loc of BH Vect File: " << pfile2 << endl;
cout<< " Size of BH Vect: " <<  bv.rows<< "x" <<bv.cols << endl;
ofstream datfile2(pfile2);
if (datfile2.is_open()) {
   cout << " Printing ph values to file:" << endl;

  for(int i = 0; i < bh.rows; i++){
     for(int j = 0; j < bh.cols; j++){
       datfile2 << bh.at<int>(i,j) << " "; 
   //for (auto i = p.begin(); i != p.end(); i++){
    //datfile << *i << endl
     }
     datfile2 << endl;
   }
}
else {
cout<<" fail to open Data file to print BH mat"<< endl;
}
if (datfile2.is_open()) datfile2.close();
cout<<" Completed writin BH to file, CLeanup" << endl;




   cout<< "Calling phogDescriptor" <<endl;

vector<double> p = vgg_phogDescriptor(bh_roi,bv_roi,Level,bin);

   cout<< "received phogDescriptor results" <<endl;


string pfile = src+".txt";
cout<< " Loc of P Vect File: " << pfile << endl;
cout<< " Size of P Vect: " << p.size() << endl;
ofstream datfile(pfile);
if (datfile.is_open()) {
   datfile << fixed << setprecision(6);
   cout << " Printing p values to file:" << endl;
  for(int i = 0; i < p.size(); i++){
     datfile << (double)p[i] << endl;
   //for (auto i = p.begin(); i != p.end(); i++){
    //datfile << *i << endl;
   }
}
else {
cout<<" fail to open Data file to print PHOG"<< endl;
}
if (datfile.is_open()) datfile.close();
cout<<" Completed writin to file, CLeanup" << endl;
//bh_roi.release();
//bv_roi.release();
//Img.release();
//p.clear();
//G.release();
//Gr.release(); 
//E.release();
//A.release();
cout<< "phogDescriptor results printed to file. Control sent back to Main " <<endl;
//return p;

}






// The main program calling all the other functions
int main (){

string src =  "/home/alex/Downloads/101_ObjectCategories/accordion/image_0001.jpg";  // "/home/alex/Pictures/SampleV4RL1.jpg";
Mat Img = imread(src); // Read the image file
imshow("Original",Img);
waitKey(0);
bh = Mat(Img.rows,Img.cols, CV_16UC1,Scalar(1));
bv = Mat(Img.rows, Img.cols, CV_32FC1,Scalar(0.0));
cout<<"Image read of Size " << Img.rows << "x" <<Img.cols <<endl;
int bin = 40;
int angle = 360; // Angle needed {180, 360}
int Level=0; // Pyramid Levels
int roi[] = {0,Img.rows-1,0,Img.cols-1}; // Region of Interest
//vector<double> p = 
vgg_phog(src, Img,bin,angle,Level,roi);
//bh.release();
//bv.release();
//Img.release();
cout<<" Completed PHOG for this Image, Thank you."<< endl;

return 0;

}
