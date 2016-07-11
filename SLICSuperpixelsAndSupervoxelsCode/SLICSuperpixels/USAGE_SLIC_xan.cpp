/*
SLIC class declaration and implementation files are provided. The files provide the code to perform superpixel segmentation as explained in the paper:

"SLIC Superpixels", Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and Sabine Susstrunk.
EPFL Technical Report no. 149300, June 2010.

The code in addition provides the extension to supervoxels as well! Usage can be seen in the demo function DoSupervoxelVideoSegmentation() in the SLICSuperpixelsDlg.cpp file.

The usage is quite straight forward. One has to instantiate an object of the SLIC class and call the various methods on it. Here is an example main() file:
*/

// TO RUN this use: g++ USAGE_SLIC_xan.cpp SLIC.cpp `pkg-config opencv --cflags --libs`


#include <string>
#include "SLICSuperpixels/SLIC.h"
#include "SLIC.h"
#include<iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <fstream>
#include <stdarg.h>
 #include <cstdio>

typedef unsigned int UINT;
using namespace cv;


const double THRESHOLD = 400;

/*
void ReadImage(unsigned int* pbuff, int* width, int* height) {
  std::string path = "/home/alex/Desktop/V4RL_MasterThesis/testImg.jpg";
  cv::Mat image = cv::imread(path);
  cv::imshow("testImg",image);
  cv::waitKey(0);

  *width = image.cols;
  *height = image.rows;
  pbuff = new UINT[(*width)*(*height)];

  for (int y=0; y<image.rows; y++) {
    for (int x=0; x<image.rows; x++) {
      cv::Vec3b p = image.at<cv::Vec3b>(y,x);
      uint8_t a = 0;
      uint8_t r = p[2];
      uint8_t g = p[1];
      uint8_t b = p[0];
      unsigned int argb = ((unsigned int)a) << 24 | ((unsigned int)r) << 16 | ((unsigned int)g) << 8 | ((unsigned int)b);
      pbuff[x+image.cols*y]=argb;
    }
  }
}
*/

cv::Mat SaveSegmentedImageFile(unsigned int* pbuff, int width, int height) {
  // write segmented image
  cv::Mat image1 = cv::Mat(height,width,CV_8UC3);
  for (int i=0; i<width*height; i++) {
    unsigned int argb = pbuff[i];
    uint8_t r = (argb >> 16) & 0x0000ff;
    uint8_t g = (argb >> 8)  & 0x0000ff;
    uint8_t b = (argb)       & 0x0000ff;
    cv::Vec3b p;
    p[0] = b;
    p[1] = g;
    p[2] = r;
    image1.at<cv::Vec3b>(i/width,i%width) = p;
  }
  return image1;
  //cv::waitKey(0);
}

Mat src, src_gray;
Mat dst, detected_edges;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map";

/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
void CannyThreshold(int, void*)
{
  /// Reduce noise with a kernel 3x3
  blur( src_gray, detected_edges, Size(3,3) );

  /// Canny detector
  Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

  /// Using Canny's output as a mask, we display our result
  dst = Scalar::all(0);

  src.copyTo( dst, detected_edges);
  imshow( window_name, dst );
 }

//Function to load label files
int** LoadLabel(std::string filel, int x, int y) {
  int** array2D = 0;
  array2D = new int*[x];
  ifstream fileIn;
  fileIn.open(filel.c_str());

  if (!fileIn) {
    cout << "Cannot open file.\n";
    return 0;
  }

  for (int i = 0; i < x; i++) {
    array2D[i] = new int[y];
    for (int j = 0; j < y; j++) {
      //if ( !(fileIn >> array2D[i][j]) ) 
      //   {
      //       std::cerr << "error while reading file";
      //       break;
      //   }
      //else {
          fileIn >> array2D[i][j];
       //  }
    }
    if ( !fileIn ) break;
  }

  fileIn.close();
  return array2D;
}

// for multiple image display

cv::Mat createOne(std::vector<cv::Mat> & images, int cols, int min_gap_size)
{
    // let's first find out the maximum dimensions
    int max_width = 0;
    int max_height = 0;
    for ( int i = 0; i < images.size(); i++) {
        // check if type is correct 
        // you could actually remove that check and convert the image 
        // in question to a specific type
        //if ( i > 0 && images[i].type() != images[i-1].type() ) {
        //    std::cerr << "WARNING:createOne failed, different types of images";
        //    return cv::Mat();
        //}
        max_height = std::max(max_height, images[i].rows);
        max_width = std::max(max_width, images[i].cols);
    }
    // number of images in y direction
    int rows = std::ceil(images.size() / cols);

    // create our result-matrix
    cv::Mat result = cv::Mat::zeros(rows*max_height + (rows-1)*min_gap_size,
                                    cols*max_width + (cols-1)*min_gap_size, images[0].type());
    size_t i = 0;
    int current_height = 0;
    int current_width = 0;
    for ( int y = 0; y < rows; y++ ) {
        for ( int x = 0; x < cols; x++ ) {
            if ( i >= images.size() ) // shouldn't happen, but let's be safe
                return result;
            // get the ROI in our result-image
            cv::Mat to(result,
                       cv::Range(current_height, current_height + images[i].rows),
                       cv::Range(current_width, current_width + images[i].cols));
            // copy the current image to the ROI
            images[i++].copyTo(to);
            current_width += max_width + min_gap_size;
        }
        // next line - reset width and update height
        current_width = 0;
        current_height += max_height + min_gap_size;
    }
    return result;
}


// for HoG visualization
Mat get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size )
{
    const int DIMX = size.width;
    const int DIMY = size.height;
    float zoomFac = 3;
    Mat visu;
    resize(color_origImg, visu, Size( (int)(color_origImg.cols*zoomFac), (int)(color_origImg.rows*zoomFac) ) );
 
    int cellSize        = 8;
    int gradientBinSize = 9;
    float radRangeForOneBin = (float)(CV_PI/(float)gradientBinSize); // dividing 180 into 9 bins, how large (in rad) is one bin?
 
    // prepare data structure: 9 orientation / gradient strenghts for each cell
    int cells_in_x_dir = DIMX / cellSize;
    int cells_in_y_dir = DIMY / cellSize;
    float*** gradientStrengths = new float**[cells_in_y_dir];
    int** cellUpdateCounter   = new int*[cells_in_y_dir];
    for (int y=0; y<cells_in_y_dir; y++)
    {
        gradientStrengths[y] = new float*[cells_in_x_dir];
        cellUpdateCounter[y] = new int[cells_in_x_dir];
        for (int x=0; x<cells_in_x_dir; x++)
        {
            gradientStrengths[y][x] = new float[gradientBinSize];
            cellUpdateCounter[y][x] = 0;
 
            for (int bin=0; bin<gradientBinSize; bin++)
                gradientStrengths[y][x][bin] = 0.0;
        }
    }
 
    // nr of blocks = nr of cells - 1
    // since there is a new block on each cell (overlapping blocks!) but the last one
    int blocks_in_x_dir = cells_in_x_dir - 1;
    int blocks_in_y_dir = cells_in_y_dir - 1;
 
    // compute gradient strengths per cell
    int descriptorDataIdx = 0;
    int cellx = 0;
    int celly = 0;
 
    for (int blockx=0; blockx<blocks_in_x_dir; blockx++)
    {
        for (int blocky=0; blocky<blocks_in_y_dir; blocky++)
        {
            // 4 cells per block ...
            for (int cellNr=0; cellNr<4; cellNr++)
            {
                // compute corresponding cell nr
                cellx = blockx;
                celly = blocky;
                if (cellNr==1) celly++;
                if (cellNr==2) cellx++;
                if (cellNr==3)
                {
                    cellx++;
                    celly++;
                }
 
                for (int bin=0; bin<gradientBinSize; bin++)
                {
                    float gradientStrength = descriptorValues[ descriptorDataIdx ];
                    descriptorDataIdx++;
 
                    gradientStrengths[celly][cellx][bin] += gradientStrength;
 
                } // for (all bins)
 
 
                // note: overlapping blocks lead to multiple updates of this sum!
                // we therefore keep track how often a cell was updated,
                // to compute average gradient strengths
                cellUpdateCounter[celly][cellx]++;
 
            } // for (all cells)
 
 
        } // for (all block x pos)
    } // for (all block y pos)
 
 
    // compute average gradient strengths
    for (celly=0; celly<cells_in_y_dir; celly++)
    {
        for (cellx=0; cellx<cells_in_x_dir; cellx++)
        {
 
            float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];
 
            // compute average gradient strenghts for each gradient bin direction
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
            }
        }
    }
 
    // draw cells
    for (celly=0; celly<cells_in_y_dir; celly++)
    {
        for (cellx=0; cellx<cells_in_x_dir; cellx++)
        {
            int drawX = cellx * cellSize;
            int drawY = celly * cellSize;
 
            int mx = drawX + cellSize/2;
            int my = drawY + cellSize/2;
 
            rectangle(visu, Point((int)(drawX*zoomFac), (int)(drawY*zoomFac)), Point((int)((drawX+cellSize)*zoomFac), (int)((drawY+cellSize)*zoomFac)), Scalar(100,100,100), 1);
 
            // draw in each cell all 9 gradient strengths
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                float currentGradStrength = gradientStrengths[celly][cellx][bin];
 
                // no line to draw?
                if (currentGradStrength==0)
                    continue;
 
                float currRad = bin * radRangeForOneBin + radRangeForOneBin/2;
 
                float dirVecX = cos( currRad );
                float dirVecY = sin( currRad );
                float maxVecLen = (float)(cellSize/2.f);
                float scale = 2.5; // just a visualization scale, to see the lines better
 
                // compute line coordinates
                float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
                float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
                float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
                float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;
 
                // draw gradient visualization
                line(visu, Point((int)(x1*zoomFac),(int)(y1*zoomFac)), Point((int)(x2*zoomFac),(int)(y2*zoomFac)), Scalar(0,255,0), 1);
 
            } // for (all bins)
 
        } // for (cellx)
    } // for (celly)
 
 
    // don't forget to free memory allocated by helper data structures!
    for (int y=0; y<cells_in_y_dir; y++)
    {
        for (int x=0; x<cells_in_x_dir; x++)
        {
            delete[] gradientStrengths[y][x];
        }
        delete[] gradientStrengths[y];
        delete[] cellUpdateCounter[y];
    }
    delete[] gradientStrengths;
    delete[] cellUpdateCounter;
 
    return visu;
 
} // get_hogdescriptor_visu


int main()
{
	int width(0), height(0);
	// unsigned int (32 bits) to hold a pixel in ARGB format as follows:
	// from left to right,
	// the first 8 bits are for the alpha channel (and are ignored)
	// the next 8 bits are for the red channel
	// the next 8 bits are for the green channel
	// the last 8 bits are for the blue channel
	// unsigned int* pbuff = new UINT[sz];
	// width=550px, height=413px
	// ReadImage(pbuff, width, height);//YOUR own function to read an image into the ARGB format
	std::string path = "/home/alex/Pictures/SampleV4RL1.jpg";
	//std::string path = "/home/alex/Desktop/V4RL_MasterThesis/testImg.jpg";
 	cv::Mat image = cv::imread(path);
  	//cv::imshow("Original Image",image);
  	//cv::waitKey(0);

  	width = image.cols;
  	height = image.rows;
  	unsigned int* pbuff = new UINT[(width)*(height)];

  	for (int y=0; y<image.rows; y++) {
  	  for (int x=0; x<image.rows; x++) {
  	    cv::Vec3b p = image.at<cv::Vec3b>(y,x);
  	    uint8_t a = 0;
 	    uint8_t r = p[2];
  	    uint8_t g = p[1];
            uint8_t b = p[0];
            unsigned int argb = ((unsigned int)a) << 24 | ((unsigned int)r) << 16 | ((unsigned int)g) << 8 | ((unsigned int)b);
            pbuff[x+image.cols*y]=argb;
  	  }
  	}

  	std::cout << "Read image of size " << width << "x" <<  height << std::endl;
	std::cout << "pbuff[0]=" << pbuff[0] << std::endl;

	//----------------------------------
	// Initialize parameters
	//----------------------------------
	int k = 200;//Desired number of superpixels.
	double m = 20;//Compactness factor. use a value ranging from 10 to 40 depending on your needs. Default is 10
	int* klabels = NULL;
	int numlabels(0);
	const string filename = "testImgout.jpg";
	const string savepath = "/home/alex/Desktop/V4RL_MasterThesis/";
	std::cout << "Done with Segmentation 1" << std::endl;
	//----------------------------------
	// Perform SLIC on the image buffer
	//----------------------------------
	SLIC segment;
	segment.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(pbuff, width, height, klabels, numlabels, k, m);
	std::cout << "Done with Segmentation 2" << std::endl;
	// Alternately one can also use the function DoSuperpixelSegmentation_ForGivenStepSize() for a desired superpixel size
	//----------------------------------
	// Save the labels to a text file
	//----------------------------------
	// segment.SaveSuperpixelLabels(klabels, width, height, filename, savepath);
	//----------------------------------
	// Draw boundaries around segments
	//----------------------------------
	segment.DrawContoursAroundSegments(pbuff, klabels, width, height, 0xff0000);
	std::cout << "Done with Segmentation 3" << std::endl;
	//----------------------------------
	// Save the image with segment boundaries.
	//----------------------------------
	Mat image1 = SaveSegmentedImageFile(pbuff, width, height);//YOUR own function to save an ARGB buffer as an image
	std::cout << "Done with Segmentation" << std::endl;
  	cv::imshow("SLIC Superpixels",image1);
  	//cv::waitKey(0);
	
	//----------------------------------
	// Clean up
	//----------------------------------
	if(pbuff) delete [] pbuff;
	if(klabels) delete [] klabels;
	
	// Read the image again
	src = imread( path );

  	/// Create a matrix of the same type and size as src (for dst)
  	dst.create( src.size(), src.type() );

	/// Convert the image to grayscale
  	cvtColor( src, src_gray, CV_BGR2GRAY );

  	/// Create a window
  	namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  	/// Create a Trackbar for user to enter threshold
  	createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );
	std::cout << "Canny Edge Processing" << std::endl;
  	/// Show the image
  	CannyThreshold(0, 0);

  	/// Wait until user exit program by pressing a key
  	//waitKey(0);


  	//-- Step 1: Detect the keypoints using SURF Detector
 	int minHessian = 100;
	// Read the image again
	src = imread( path );
	/// Convert the image to grayscale
  	cvtColor( src, src_gray, CV_BGR2GRAY );
  	Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create( minHessian );

  	std::vector<KeyPoint> keypoints_1;
  	detector->detect( src_gray, keypoints_1 );
	Mat img_keypoints_1;
	drawKeypoints( src_gray, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
	std::cout << " Value of SURF keypoint feature: "<< img_keypoints_1.at<double>(0,0) << std::endl;
	//imshow("SURF", img_keypoints_1 );
	std::cout << "Done with Segmentation 1" << std::endl;
	// Step 2:  get intensity at the SURF keypoints
	vector<uchar> intensity(keypoints_1.size());
	for(int i=0;i<keypoints_1.size();i++){
	 	intensity[i] = src_gray.at<uchar>(keypoints_1[i].pt);
	}
	// step 3: get class at the SURF keypoints
	std::string path1 = "/home/alex/Desktop/V4RL_MasterThesis/iccv09Data/labels/9004879.regions.txt";
	//int** lab2D = LoadLabel(path1,src_gray.rows, src_gray.cols);
	//vector<int> classLab(keypoints_1.size());
	//for(int i=0;i<keypoints_1.size();i++){
	// 	classLab[i] = lab2D[(int)keypoints_1[i].pt.x][(int)keypoints_1[i].pt.y];
	//}
  	
	//waitKey(0);
	// Step 1: detect the keypoints using SIFT detectors
  	Ptr<cv::xfeatures2d::SIFT> detector1 = cv::xfeatures2d::SIFT::create( minHessian );
  	std::vector<KeyPoint> keypoints_2;
  	detector1->detect( src_gray, keypoints_2 );
	Mat img_keypoints_2;
	drawKeypoints( src_gray, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
	//imshow("SIFT", img_keypoints_2 );
  
	//waitKey(0);

		
	// for HoG (Hist of Oriented Gradients)
	Mat imageResized;
	resize(src_gray, imageResized, Size(64,128) );
	HOGDescriptor d; //( Size(32,32), Size(8,8), Size(4,4), Size(4,4), 9);
  	vector< float> descriptorsValues;
  	vector< Point> locations;
  	d.compute( imageResized, descriptorsValues, Size(0,0), Size(0,0), locations);
	cout << "Found " << descriptorsValues.size() << " descriptor values" << endl;
	cout << "The number of locations for HoG: " << locations.size() << endl;
	//Mat img_keypoints_3 = cv::Mat(imageResized.rows,imageResized.cols,CV_16UC1);;
	Mat img_keypoints_3 = get_hogdescriptor_visu(src_gray, descriptorsValues, Size(src_gray.rows,src_gray.cols));
	for(int i =0; i<locations.size();i++){
		img_keypoints_3.at<float>(locations[i]) = descriptorsValues[i];
	}

  	//show image
  	//imshow("HoG Values", img_keypoints_3);
	//waitKey(0);

	// show all images in a Big window:
	vector<Mat> bigImage;
	bigImage.push_back(image);
	bigImage.push_back(dst);
	bigImage.push_back(image1);
	bigImage.push_back(img_keypoints_1);
	bigImage.push_back(img_keypoints_2);
	//bigImage.push_back(img_keypoints_3);
	
	Mat bigImage1 = createOne(bigImage,3,2);
  	imshow("All Values", bigImage1);
  	waitKey(0);
	
	// Release all memory used
	image1.release();
	dst.release();
	image.release();
	src.release();
	src_gray.release();
	img_keypoints_1.release();
	img_keypoints_2.release();
	keypoints_1.clear();
	keypoints_2.clear();
	locations.clear();
	descriptorsValues.clear();
	img_keypoints_3.release();
	bigImage.clear();
	
	
	
	
	//cvShowManyImages("Images", 5, image, image1, dst, img_keypoints_1, img_keypoints_2);
	//waitKey(0);

	return 0;
}

