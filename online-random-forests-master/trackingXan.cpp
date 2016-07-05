#include<iostream>
#include<fstream>
#include<vector>
#include<string>
#include<bzlib.h>
#include<sstream>

using namespace std;

int main(){
  ifstream fileList("/home/alex/Downloads/caltech/splits/trainfiles.csv");
  string testarr, j;
  ifstream labelList("/home/alex/Downloads/caltech/splits/trainLabelN1.csv");
  
  if(fileList.is_open() && labelList.is_open())
  {
  //file opened successfully so we are here
    cout << "Files Opened successfully!!!. Reading data from file into array" << endl;
    //this loop run until end of file (eof) does not occur
  while(getline(fileList, testarr))
  {
  string fname = "/home/alex/Downloads/caltech101_features_PHOG/phog/A360_K40/Level2/"+testarr+".bz2";
  //cout << fname << endl;
  FILE *f1;
  int bzError;
  BZFILE *bzf;
  char buf[4096];
  f1 = fopen(fname.c_str(), "r");
  bzf = BZ2_bzReadOpen(&bzError, f1, 0, 0, NULL, 0);
  if (bzError != BZ_OK) {
    fprintf(stderr, "E: BZ2_bzReadOpen: %d\n", bzError);
    return -1;
  }
  while (bzError == BZ_OK) {
    int nread = BZ2_bzRead(&bzError, bzf, buf, sizeof buf);
    if (bzError == BZ_OK || bzError == BZ_STREAM_END) {
      //size_t nwritten = fwrite(buf, 1, nread, stdout);
      istringstream ss( buf );
      int i=1;
      ofstream testPHOG("/home/alex/Downloads/caltech101_features_PHOG/phog/A360_K40/Level2/trainPHOG_N1.txt", ios::app);
      if (testPHOG.is_open()){
      getline(labelList, j);
      testPHOG << j;
      while (ss)
      {
        string s;
        if (!getline( ss, s, ',' )) break;
        testPHOG << " " << i << ":" << s ;
        i++;
      }
      testPHOG << endl;
      testPHOG.close();
      }
      else{
       cout<< " couldnot open PHOG Test file to write" << endl;
      }
      //if (nwritten != (size_t) nread) {
      //  fprintf(stderr, "E: short write\n");
      //  return -1;
      //}
    }
  }

  if (bzError != BZ_STREAM_END) {
    fprintf(stderr, "E: bzip error after read: %d\n", bzError);
    return -1;
  }

  BZ2_bzReadClose(&bzError, bzf);
  fclose(f1);
  }
  fileList.close();
  }
  else //file could not be opened
	{
          cout << "File could not be opened." << endl;
	}

  

  return 0;
}

