// test3.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\core\core.hpp>
#include <conio.h>
#include <math.h>
#include <stack>
#include <queue>
#include <string.h>

using namespace cv;
using namespace std;
/*void displayg(long int a[])
{
	for (int i = 0; i < 26; i++)
	{
		for (int j = 0; j < a[i]; j++)
		{
			printf("*");
		}
		printf("\n");
	}
}
int main()
{
	Mat im = imread("C:\\Users\\Shakul Pathak.000\\Desktop\\IP\\lena10.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	if (im.empty())
	{

		cout << "Cannot load image!" << endl;
		return -1;

	}
	long int a[26];
	for (int i = 0; i < 26; i++)
	{
		a[i] = 0;
	}
	long int a0 = 0;
	for (int i = 0; i < im.rows; i++)
	{
		for (int j = 0; j < im.cols; j++)
		{
			for (int k = 0; k < 260; k += 10)
			{
				if (k <= (im.at<uchar>(i, j)) && (im.at<uchar>(i, j)) < (k + 10))
				{
					a[a0++] += 1;
				}
			}


		}

	}
	displayg(a);
	
	getchar();
} */
/*
int main()
{
	Mat x(1200, 1200, CV_8UC1);
	int xc = 600;
	int yc = 600;
	float rc =600;
	for (int i = 0; i < 600; i++)
	{
		for (int j = 0; j < 600; j++)
		{
			x.at<uchar>(i, j) = 0;
		}
	}
	float k = rc*rc;
	for (; k >= 0; k--)
	{
		for (int i = 0; i < 600; i++)
		{
			for (int j = 0; j < 600; j++)
			{
				if ((i - xc)*(i - xc) + (j - yc)*(j - yc) <= (rc*rc - k))
				{
					x.at<uchar>(i, j) = (int)(255 * (((float)k / (float)rc*rc)));

				}
			}

		}
	}
	imshow("final", x);
	waitKey(0);
}
*/
/*int main()
{
	int a = 0;
	int b = 0;
	Mat im = imread("C:\\Users\\Shakul Pathak.000\\Desktop\\IP\\lena10.jpg");
	if (im.empty()){

		cout << "Cannot load image!" << endl;
		return -1;

	}
	Mat x(2*im.rows, 2 * im.cols, CV_8UC3);
	for (int i = 0; i < 2*im.rows; i++)
	{
		for (int j = 0; j < 2*im.cols; j++)
		{
			
				x.at<Vec3b>(i, j)[0] = im.at<Vec3b>(i/2, j/2)[0];
				x.at<Vec3b>(i, j)[1] = im.at<Vec3b>(i/2, j/2)[1];
				x.at<Vec3b>(i, j)[2] = im.at<Vec3b>(i/2, j/2)[2];
		
			}
			
			
		}
		

	imshow("lena", x);
	waitKey(0);
}
*/
/*int main()
{
	Mat x(800, 800, CV_8UC1);
	long int k;
	int m;
	namedWindow("A", WINDOW_NORMAL);
	createTrackbar("T", "A", &m, 3200);
	char ch;
	while (1)
	{
		if (m != 0)
		{
			for (int i = 0; i < 800; i++)
			{
				for (int j = 0; j < 800; j++)
				{
					k = (i - 400)*(i - 400) + (j - 400)*(j - 400);
					k = k / m;
					if (k>255)
						x.at<uchar>(i, j) = 0;
					else
						x.at<uchar>(i, j) = 255 - k;

				}
			}
		}
		imshow("A", x);

		ch = waitKey(0);
		if (ch == '0')
			break;
	}
	waitKey(0);
} */
/*int main()
{
	Mat im = imread("C:\\Users\\Shakul Pathak.000\\Desktop\\IP\\lena10.jpg");
	if (im.empty()){

		cout << "Cannot load image!" << endl;
		return -1;

	}
	Mat x = im;
	for (int i = 0; i < im.rows; i++)
	{
		for (int j = 0; j < im.cols; j++)
		{
			im.at<Vec3b>(i/2, j/2) = x.at<Vec3b>(i , j );
		}
	}
	im.rows = im.rows / 2;
	im.cols = im.cols / 2;

	imshow("A", im);
	waitKey(0);
}
*/
/*
int main()
{
	Mat im = imread("C:\\Users\\Shakul Pathak.000\\Desktop\\IP\\lena10.jpg");
	if (im.empty()){

		cout << "Cannot load image!" << endl;
		return -1;

	}
	int m;
	for (int i = 1; i < im.rows - 1; i++)
	{
		for (int j = 1; j < im.cols - 1; j++)
		{
			for (m = 0; m < 3; m++)
			{
				im.at<Vec3b>(i, j)[m] = ((im.at<Vec3b>(i - 1, j)[m]) + (im.at<Vec3b>(i, j)[m]) + (im.at<Vec3b>(i + 1, j)[m]) + (im.at<Vec3b>(i + 1, j + 1)[m]) +( im.at<Vec3b>(i - 1, j - 1)[m]) + (im.at<Vec3b>(i, j + 1)[m]) + (im.at<Vec3b>(i, j - 1)[m]) + (im.at<Vec3b>(i - 1, j + 1)[m]) + (im.at<Vec3b>(i + 1, j - 1)[m]))/9;
			}
		}
	}
	
	imshow("A", im);
	waitKey(0);

} */
/*int main()
{
	/*Mat im = imread("C:\\Users\\Shakul Pathak.000\\Desktop\\IP\\lena10.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat x(im.rows, im.cols, CV_8UC1);
	double  d;
	for (int i = 0; i < im.rows; i++)
	{
		for (int j = 0; j < im.cols; j++)
		{
			d = (i - (im.rows/2))*(i - (im.rows/2)) + (j-(im.cols/2))*(j-(im.cols/2));
			x.at<uchar>(i, j) = 255*(1-(d/(256*256*2)));
			im.at<uchar>(i, j) =(int) im.at<uchar>(i, j)*pow(((float)x.at<uchar>(i, j)/(float)255),2.5);
		}
	}
	imshow("v", x);
	imshow("x0", im); */
	/*fn();
	waitKey(0);

}  */
/*int main()
{
	Mat im = imread("C:\\Users\\Shakul Pathak.000\\Desktop\\IP\\lena10.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	if (im.empty()){

		cout << "Cannot load image!" << endl;
		return -1;

	}
	int m;
	imshow("B", im);
	for (int i = 0; i < im.rows; i++)
	{
		for (int j = 0; j < im.cols; j++)
		{
			int c = 0; int a[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
			m = 0;
			for (int k = 1; k >= -1; k--)
			{
				for (int l = -1; l <= 1; l++)
				{ 
					if (i >= 1 && j >= 1 && i < im.rows - 1 && j < im.cols - 1)
					{
						a[m] = im.at<uchar>(i + k, j + l);
						m++;
					}
					
				}
			}
			sort(a, a + 8);
			im.at<uchar>(i, j) = a[m/2];
		}
	}
	imshow("A", im);
	waitKey(0);
}
*/
/*int main()
{
	double t;
	Mat im = imread("C:\\Users\\Shakul Pathak.000\\Desktop\\IP\\lena10.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat x = im;
	
	for (int i = 0; i < im.rows; i++)
	{
		for (int j = 0; j < im.cols; j++)
		{
			if ((int)(i*(0.5) + j*(1.73 / 2))<im.rows && (int)(-(1.73 / 2)*i, j*(0.5))<im.cols)
			x.at<uchar>(i, j) = im.at<uchar>((int)(i*0.5+j*(1.73/2) ), (int)( -(1.73/2)*i,j*(0.5)));
		}
	}
	imshow("Rot", x);
	waitKey(0);

} */
Mat im = imread("C:\\Users\\Shakul Pathak.000\\Desktop\\IP\\S.jpg");
Mat mark(im.rows, im.cols, CV_8UC1, Scalar(0));
Mat mark2(im.rows, im.cols, CV_8UC1, Scalar(0));
int col = 1;
int f=0;
int check(int i, int j)
{
	if (im.at<Vec3b>(i, j)[0] != 255 || im.at<Vec3b>(i, j)[1] != 255 || im.at<Vec3b>(i, j)[2] != 255)
		return 1;
	else
		return 0;
}
struct node
{
	int x;
	int y;
};
typedef struct node node;
void shift(int t, void* b)
{
	Mat mark3(im.rows, im.cols, CV_8UC1, Scalar(0));
	for (int i = 0; i <mark.rows; i++)
	{
		for (int j = 0; j<mark.cols; j++)
		{
			
		 if (mark.at<uchar>(i, j) == 255)
			{
				
				if ((j + t) >=(mark.cols))
				{
					if ((j+t-mark.cols)<mark.cols&&(t+j-mark.cols)>=0)
					mark3.at<uchar>(i , (j+t-mark.cols)) = mark.at<uchar>(i, j);
					
				}
				else
					mark3.at<uchar>(i, j+t) = mark.at<uchar>(i, j);

			}
		}
	}
	imshow("a", mark3);
}
/*
void dfs(int i, int j)
{
	mark.at<uchar>(i, j) = 255;

for (int k = -1; k <= 1; k++)
{
for (int l = -1; l <= 1; l++)
{
if (mark.at<uchar>(i+l,j+k) == 0&&((l)!=0)&&((k)!=0))
if (check(i+l,j+k))
dfs(i + l, j + k);
}
}
} 
int check3(int i, int j)
{
	for (int k = -1; k <= 1; k++)
	{
		for (int l = -1; l <= 1; l++)
		{
			if (mark.at<uchar>(i + l, j + k) >0  && (l != 0) && (k != 0))
			if (check(i+l,j+k))
				return 1;
		}
	}
	return 0;
} */
/*queue <node> s;
void bfs(int i, int j)
{
	mark.at<uchar>(i, j) = 255;
	for (int k = -1; k <= 1; k++)
	{
		for (int l = -1; l <= 1; l++)
		{
			if (mark.at<uchar>(i + l, j + k) == 0 && (l != 0) && (k != 0))
			{
				mark.at<uchar>(i + l, j + k) = 255;
				node n;
				n.x = (i + l);
				n.y = (k + j);
				s.push(n);
			}
		}
		while (!s.empty())
		{
			node n = s.front();
			mark.at<uchar>(n.x, n.y) = 255;
			s.pop();
		}
	}
}

int main()
{
	for (int i = 1; i < im.rows - 1; i++)
	{
		for (int j = 1; j < im.cols - 1; j++)
		{
			if (check(i, j) && mark.at<uchar>(i, j) == 0)
			{
				bfs(i, j);

			}
		
			}
	}

	mark2 = mark;
	int t = 1;
	int d = 0;
	namedWindow("a");
	createTrackbar("T", "a", &t,mark.cols,shift,NULL);
	
imshow("a", mark);
	waitKey(0);
	return 0;

}  */

/*
Mat take[100];
void mirror(Mat x)
{
	for (int i = 0; i < x.rows; i++)
	{
		for (int j = 0; j < x.cols/2; j++)
		{
			x.at<Vec3b>(i, j) = x.at<Vec3b>(i, (x.cols / 2) - j);
		}
	}
}
int main()
{
	VideoCapture stream1(0);   

	if (!stream1.isOpened()) { 
		cout << "cannot open camera";
	}
	int i = 0;
	
	while (i<10) {
		Mat cameraFrame;
		stream1.read(cameraFrame); 
		mirror(cameraFrame);
		imshow("cam", cameraFrame);
		take[i] = cameraFrame;
		i++;
		if (waitKey(30) >= 0)
			break;
	}
	char str[200] = "C:\\Users\\Shakul Pathak.000\\Desktop\\IP\\";
	char str2[200] = "C:\\Users\\Shakul Pathak.000\\Desktop\\IP\\";
	char s[20] = "a";
	char j[10] = ".jpg";
	for (int i = 0; i < 10; i++)
	{
		strcat_s(str, s);
		strcat_s(str, j);
		imwrite(str, take[i]);
		s[0]++;
		strcpy_s(str, str2);
	}

	waitKey(0);
	return 0;
}
*/
/*int main(int argc, char** argv)
{
	VideoCapture cap("C:\\Users\\Shakul Pathak.000\\Desktop\\IP\\new.mp4"); 
	Mat imgTmp;
	cap.read(imgTmp);
	if (!cap.isOpened())  
	{
		cout << "Cannot open" << endl;
		return -1;
	}
namedWindow("Control Unit", CV_WINDOW_NORMAL); 
int iLH = 14;
int iHH = 35;
int iLS = 16;
int iHS = 107;
int iLV = 237;
int iHV = 255;
int Lth = 0;
int Hth = 255;
int xx = -1;
int yy = -1;
cvCreateTrackbar("LowH", "Control Unit", &iLH, 179);     cvCreateTrackbar("HighH", "Control Unit", &iHH, 179);
cvCreateTrackbar("LowS", "Control Unit", &iLS, 255);   cvCreateTrackbar("HighS", "Control Unit", &iHS, 255);
cvCreateTrackbar("LowV", "Control Unit", &iLV, 255);     cvCreateTrackbar("HighV", "Control Unit", &iHV, 255);
cvCreateTrackbar("Lth", "Control Unit", &Lth, 255); cvCreateTrackbar("Hth", "Control Unit", &Hth, 255);
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
while (1)
	{
	Mat img;
bool b = cap.read(img); 
if (!b) 
{
	break;
}
Mat imgHSV;
cvtColor(img, imgHSV, COLOR_BGR2HSV); 
Mat imgThresh;

inRange(imgHSV, Scalar(iLH, iLS, iLV), Scalar(iHH, iHS, iHV), imgThresh); 

erode(imgThresh, imgThresh, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
dilate(imgThresh, imgThresh, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

dilate(imgThresh, imgThresh, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
erode(imgThresh, imgThresh, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));  
erode(imgThresh, imgThresh, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
dilate(imgThresh, imgThresh, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

dilate(imgThresh, imgThresh, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
erode(imgThresh, imgThresh, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
Canny(imgThresh, imgThresh, Lth, Hth);
/*Moments oMoments = moments(imgThresh);

double dM01 = oMoments.m01;
double dM10 = oMoments.m10;
double dArea = oMoments.m00;

// if the area <= 10000, I consider that the there are no object in the image and it's because of the noise, the area is not zero 
if (dArea > 10000)
{
	//calculate the position of the ball
	int XX = dM10 / dArea;
	int YY = dM01 / dArea;

	if (xx >= 0 && yy >= 0 && XX >= 0 && YY >= 0)
	{
		//Draw a red line from the previous point to the current point
		line(imgLines, Point(XX, YY), Point(xx, yy), Scalar(255, 0, 0), 2);
	}

	xx = XX;
	yy = YY;
}
*/
/*imshow("Thresholded Image", imgThresh); 
/* imshow("Original", img);  */

	/*	if (waitKey(30) == 27) 
		{
			cout << "esc key!" << endl;
			break;
		}
	}

	return 0;

} */
int c = 0;
int g=0;
/*int check4(Mat x1)
{
	int t; int f1 = 0;
	
		for (int i = x1.rows*(0.4); i < x1.rows*(0.75); i++)
		{

			for (int j = 0; j < x1.cols / 15; j++)

			if ((x1.at<Vec3b>(i, j)[0] == 255))
			{
				return 1;
				
			}

		}
		return 0;
	
} */
int main()
{
	int iLH = 21;
	int iHH = 29;
	int iLS = 71;
	int iHS = 255;
	int iLV = 253;
	int iHV = 255;
	int Lth = 0;
	int Hth = 255;
	int t1 = 15000;
	int xx = -1;
	int yy = -1;
	Mat im2;
	double A;
	namedWindow("Control Unit", CV_WINDOW_NORMAL);
	vector<Mat> channels;
	vector<Vec4i> lines;
	vector<Vec3f> circles;
	VideoCapture cap("C:\\Users\\Shakul Pathak.000\\Desktop\\IP\\new.mp4");
	Mat imgTmp;
	cap.read(imgTmp);
	if (!cap.isOpened())
	{
		cout << "Cannot open" << endl;
		return -1;
	}
	Mat imHSV;
	Mat imH,imS,imV;

	cvCreateTrackbar("LowH", "Control Unit", &iLH, 179);     cvCreateTrackbar("HighH", "Control Unit", &iHH, 179);
	cvCreateTrackbar("LowS", "Control Unit", &iLS, 255);   cvCreateTrackbar("HighS", "Control Unit", &iHS, 255);
	cvCreateTrackbar("LowV", "Control Unit", &iLV, 255);     cvCreateTrackbar("HighV", "Control Unit", &iHV, 255);
	cvCreateTrackbar("Lth", "Control Unit", &Lth, 255); cvCreateTrackbar("Hth", "Control Unit", &Hth, 255);
	cvCreateTrackbar("Area!", "Control Unit", &t1, 50000);

	Mat imgLines = Mat::zeros(imgTmp.size(), CV_8UC3);
	int f = 0;
	while (1)
	{ 
		Mat im;		Mat im1;

		Mat imgThresh1;
		Mat imgThresh2;
		Mat imgThresh3;
		bool b = cap.read(im);

		im1 = im;

		if (!b)
			break;
		cvtColor(im, imHSV, COLOR_BGR2HSV);

		/*split(imHSV, channels);
		imH = channels[0];
		imS = channels[1];
		imV = channels[2]; */
		
		inRange(imHSV, Scalar(iLH, 0, 0), Scalar(iHH, 255, 255), imgThresh1);
		inRange(imHSV, Scalar(0, iLS, 0), Scalar(179, iHS, 255), imgThresh2);
		inRange(imHSV, Scalar(0, 0, iLV), Scalar(179, 255, iHV), imgThresh3);
		/*dilate(imgThresh1, imgThresh1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));*/
		Mat imgThresh(imgThresh1.rows, imgThresh1.cols, CV_8U);
		/*imshow("H", imgThresh1);
		imshow("S", imgThresh2);
		imshow("V",imgThresh3); */

		for (int i = 0; i < imgThresh1.rows; i++)
		{
			for (int j = 0; j < imgThresh1.cols; j++)
			{
				if ((imgThresh1.at<uchar>(i, j) == 255) && (imgThresh2.at<uchar>(i, j) == 255) && (imgThresh3.at<uchar>(i, j) == 255))
					imgThresh.at<uchar>(i,j) = 255;
				else
					imgThresh.at<uchar>(i, j) = 0;
			}
		} 

	/*	Mat x2(imgThresh.rows, imgThresh.cols, CV_8UC1);

		HoughLinesP(imgThresh, lines, 1, CV_PI / 180, 5, 2, 0);
		for (size_t i = 0; i < lines.size(); i++)
		{
			Vec4i l = lines[i];
			line(x2, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255), 1, CV_AA);
		}
		imshow("final", x2); */
	/*erode(imgThresh, imgThresh, getStructuringElement(MORPH_ELLIPSE, Size(5, 5))); */
/*	dilate(imgThresh1, imgThresh1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));*/
/*	GaussianBlur(imV, imV, Size(9, 9), 2, 2);
	HoughCircles(imV, circles, CV_HOUGH_GRADIENT, 1, imV.rows / 8, 200, 100, 0, 0);
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		circle(imgThresh, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		// circle outline
		circle(imgThresh, center, radius, Scalar(0, 0, 255), 3, 8, 0);
	}
	*/
		imgLines = Mat::zeros(imgTmp.size(), CV_8UC3);

		Moments oMoments = moments(imgThresh);

		double dM01 = oMoments.m01;
		double dM10 = oMoments.m10;
		double dArea = oMoments.m00;

	/*	imgLines = Mat::zeros(imgTmp.size(), CV_8UC3); */

		if (dArea > t1)
		{
			//calculate the position of the ball
			int XX = dM10 / dArea;
			int YY = dM01 / dArea;

			if (xx >= 0 && yy >= 0 && XX >= 0 && YY >= 0)
			{
				line(imgLines, Point(XX, YY), Point(xx, yy), Scalar(255, 0,0), 1);
			}

			xx = XX;
			yy = YY;
		}
		/*imshow("Thresholded Image", imgThresh);*/
		/*for (int i = imgLines.rows*(0.4); i < imgLines.rows*(0.75); i++)
		{  
			for (int j = 0; j < imgLines.cols/18; j++)
			{
				if ((imgLines.at<Vec3b>(i, j)[0] == 255))
				{
					imgLines.at<Vec3b>(i, j)[0] = 0;
				}
			}
		} */
		f = 0;
		for (int i = 0; i < imgLines.rows; i++)
		{
			imgLines.at<Vec3b>(i, imgLines.cols/72)[2] = 255;
			if (imgLines.at<Vec3b>(i, imgLines.cols / 72)[0] == 255)
			{
				if (f==0)
				c++;
				f++;
				for (int i = 0; i < 10; i++)
				{
					A = cap.get(CV_CAP_PROP_POS_FRAMES);
					bool b = cap.read(imgTmp);
					if (b == 0)
					{
						cap.set(CV_CAP_PROP_POS_FRAMES, A);
						break;
					}
					

				}

			}
		}

		im2 = imgLines;
		im1 = im1 + imgLines;
		imshow("Original", im1); 
		imshow("final", imgThresh);
		if (waitKey(30) == '0')
			break;

	}
	cout << "Goals :" << (c/2);
	if ((c / 2) == 4)
	{
		Mat y = imread("C:\\Users\\Shakul Pathak.000\\Desktop\\IP\\3.jpg");
		imshow("SCORE", y);
	}
	getchar();
} 
/*struct p
{
	int x;
	int y;
};
int main()
{
	int iLH = 0;
	int iHH = 24;
	int iLS = 42;
	int iHS = 115;
	int iLV = 103;
	int iHV = 184;
	int Lth = 0;
	int Hth = 255;
	namedWindow("Control Unit", CV_WINDOW_AUTOSIZE);
	vector<Mat> channels;
	vector<Vec4i> lines;
	VideoCapture cap(0);
	if (!cap.isOpened())
	{
		cout << "Cannot open" << endl;
		return -1;
	}
	Mat imHSV;
	Mat imH, imS, imV;
	cvCreateTrackbar("LowH", "Control Unit", &iLH, 179);     cvCreateTrackbar("HighH", "Control Unit", &iHH, 179);
	cvCreateTrackbar("LowS", "Control Unit", &iLS, 255);   cvCreateTrackbar("HighS", "Control Unit", &iHS, 255);
	cvCreateTrackbar("LowV", "Control Unit", &iLV, 255);     cvCreateTrackbar("HighV", "Control Unit", &iHV, 255);
	cvCreateTrackbar("Lth", "Control Unit", &Lth, 255); cvCreateTrackbar("Hth", "Control Unit", &Hth, 255);
	while (1)
	{
		Mat im;
		Mat imgThresh1;
		Mat imgThresh2;
		Mat imgThresh3;

		bool b = cap.read(im);
		if (!b)
			break;
		cvtColor(im, imHSV, COLOR_BGR2HSV);

		split(imHSV, channels);
		imH = channels[0];
		imS = channels[1];
		imV = channels[2];

		inRange(imHSV, Scalar(iLH, 0, 0), Scalar(iHH, 255, 255), imgThresh1);
		inRange(imHSV, Scalar(0, iLS, 0), Scalar(179, iHS, 255), imgThresh2);
		inRange(imHSV, Scalar(0, 0, iLV), Scalar(179, 255, iHV), imgThresh3);
		Mat imgThresh(imgThresh1.rows, imgThresh1.cols, CV_8U);
		imshow("H", imgThresh1);
		imshow("S", imgThresh2);
		imshow("V", imgThresh3);

		for (int i = 0; i < imgThresh1.rows; i++)
		{
			for (int j = 0; j < imgThresh1.cols; j++)
			{
				if ((imgThresh1.at<uchar>(i, j) == 255) && (imgThresh2.at<uchar>(i, j) == 255) && (imgThresh3.at<uchar>(i, j) == 255))
					imgThresh.at<uchar>(i, j) = 255;
				else
					imgThresh.at<uchar>(i, j) = 0;
			}
		}



		erode(imgThresh, imgThresh, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		dilate(imgThresh, imgThresh, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));


		Canny(imgThresh, imgThresh, Lth, Hth);
		Mat x2(imgThresh.rows, imgThresh.cols, CV_8UC1);

		HoughLinesP(imgThresh, lines, 1, CV_PI / 180, 20, 5, 5);
		for (size_t i = 0; i < lines.size(); i++)
		{
			Vec4i l = lines[i];
			line(x2, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255), 1, CV_AA);
		}
		erode(x2, x2, getStructuringElement(MORPH_ELLIPSE, Size(1, 1)));
		int c = 0;
		struct p point;
		for (int i = 0; i < x2.rows; i++)
		{
			for (int j = 0; j < x2.cols; j++)

			{
				if (x2.at<uchar>(i, j) < 220)
					x2.at<uchar>(i, j) = 0;
				if (x2.at<uchar>(i, j)>220&&c<1)
				{
					c++;
				
					point.x = i;
					point.y = j;

				}

			}

		}
		imshow("final", x2);
			cout<< point.x<<",";
			cout << point.y << endl; */


		/*imshow("final", imgThresh); */


	/*	if (waitKey(30) == 27)
			break;

	}

} */
