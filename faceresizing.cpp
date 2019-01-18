#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <sstream>


using namespace cv;
using namespace cv::face;
using namespace std;



int main (){

    String face_cascade_name = "haar.xml";
    CascadeClassifier haar_cascade;
    haar_cascade.load(face_cascade_name);

    int input_type;
    char* input_name;
    cout<<"Single or batch? 1 = single, 2 = batch: ";
    cin>> input_type;

    /*
    cout<<"Person: ";
    cin>> input_name;
    */

    if(input_type==2){
        for(int i=1;i<=10;i++){
            string path = format("/home/chisiong/Desktop/testfolder/withdb/REALDATA/Jeff/%d.jpg", i);
            cout<< path<< endl;
            Mat original = imread(path);
            Mat gray;
            cvtColor(original, gray, COLOR_BGR2GRAY);
            vector<Rect> faces;
            haar_cascade.detectMultiScale(gray, faces, 1.2, 3);

            // 
            // if it found something, resize the found region:
            //

            if (faces.size() > 0)
            {
                Rect roi = faces[0];
                Mat cropped;
                resize(gray(roi), cropped, Size(150,150));
                string completed = format ("/home/chisiong/Desktop/testfolder/withdb/DONE/Jeff/%d.png",i);
                imwrite(completed, cropped);
                cout<< completed <<endl;
            }
        }
    }else{
        int i;
        cout << "Photo number: ";
        cin>> i;
        string path = format("/home/chisiong/Desktop/testfolder/withdb/REALDATA/Jeff/%d.jpg", i);
        cout<< path<< endl;
        Mat original = imread(path);
        Mat gray;
        cvtColor(original, gray, COLOR_BGR2GRAY);
        vector<Rect> faces;
        haar_cascade.detectMultiScale(gray, faces, 1.2, 3);

            // 
            // if it found something, resize the found region:
            //

        if (faces.size() > 0)
        {
            Rect roi = faces[0];
            Mat cropped;
            resize(gray(roi), cropped, Size(150,150));
            string completed = format ("/home/chisiong/Desktop/testfolder/withdb/DONE/Jeff/%d.png",i);
            imwrite(completed, cropped);
            cout<< completed <<endl;
        }
    }
}