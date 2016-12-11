#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\core\core.hpp>
#include <conio.h>
#include <math.h>

using namespace cv;
using namespace std;
void fn()
{
	Mat i = imread("C:\\Users\\Shakul Pathak.000\\Desktop\\IP\\lena10.jpg");
	imshow("xy",i);
}