#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <chrono>
#include<opencv2/core/core.hpp>

#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include "ImageConverter.h"
#include "BaseTrainer.h"
#include "Network.h"

using namespace cv;
using namespace std;

void xor();

int main()
{
	srand(time(NULL));
	ImageConverter converter;
	Mat traingImg = imread("training_chars.png");
	converter.prepareSamples(traingImg);
	cv::Mat img1 = cv::imread("test.jpg");
	cv::resize(img1, img1, cv::Size(), 2, 2);
	//Detect
	std::vector<cv::Rect> letterBBoxes1 = converter.detectLetters(img1);
	//Display
	for (int i = 0; i< letterBBoxes1.size(); i++)
		cv::rectangle(img1, letterBBoxes1[i], cv::Scalar(182, 255, 255), 3, 8, 0);
	using namespace std::chrono;
	milliseconds ms = duration_cast< milliseconds >(
		system_clock::now().time_since_epoch()
		);
	cv::imwrite("ConvertedImages/" + to_string(ms.count()) + ".jpg", img1);
	
	//Mat part = img1(letterBBoxes1[0]);
	//cv::imwrite("ConvertedImages/ttt2.jpg", part);

	
	/*Mat large = imread("test.jpg");
	Mat rgb;
	// downsample and use it for processing
	pyrDown(large, rgb);
	Mat small;
	cvtColor(rgb, small, CV_BGR2GRAY);
	// morphological gradient
	Mat grad;
	Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	morphologyEx(small, grad, MORPH_GRADIENT, morphKernel);
	// binarize
	Mat bw;
	threshold(grad, bw, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);
	// connect horizontally oriented regions
	Mat connected;
	morphKernel = getStructuringElement(MORPH_RECT, Size(9, 1));
	morphologyEx(bw, connected, MORPH_CLOSE, morphKernel);
	// find contours
	Mat mask = Mat::zeros(bw.size(), CV_8UC1);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(connected, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	// filter contours
	for (int idx = 0; idx >= 0; idx = hierarchy[idx][0])
	{
		Rect rect = boundingRect(contours[idx]);
		Mat maskROI(mask, rect);
		maskROI = Scalar(0, 0, 0);
		// fill the contour
		drawContours(mask, contours, idx, Scalar(255, 255, 255), CV_FILLED);
		// ratio of non-zero pixels in the filled region
		double r = (double)countNonZero(maskROI) / (rect.width*rect.height);

		if (r > .45 &&(rect.height > 8 && rect.width > 8))
		{
			rectangle(rgb, rect, Scalar(0, 255, 0), 2);
		}
	}
	imwrite("" + string("rgb.jpg"), rgb);*/

	//xor ();
	

	system("pause");
	return 0;
}

void xor()
{
	vector<double> vec{ 2, 4, 1 };
	Network network(vec);
	//network.initializeInputs(input, 0);
	//network.feedForward();

	//xor
	vector<vector<double>> neurons = { { 0,0 },{ 0,1 },{ 1,0 },{ 1,1 } };
	vector<double> targetVal = { 0,1,1,0 };

	BaseTrainer trainer(neurons, targetVal, network);
	vector<double> inp = { 0,1 };
	network.initializeInputs(inp, 0);
	network.feedForward();
	cout << "\nTest : " << inp[0] << " " << inp[1] << "\t" << network.layers.back().back().getOutputValue();

	inp = { 1,0 };
	network.initializeInputs(inp, 0);
	network.feedForward();
	cout << "\nTest : " << inp[0] << " " << inp[1] << "\t" << network.layers.back().back().getOutputValue();


	inp = { 0,0 };
	network.initializeInputs(inp, 0);
	network.feedForward();
	cout << "\nTest : " << inp[0] << " " << inp[1] << "\t" << network.layers.back().back().getOutputValue();


	inp = { 1,1 };
	network.initializeInputs(inp, 0);
	network.feedForward();
	cout << "\nTest : " << inp[0] << " " << inp[1] << "\t" << network.layers.back().back().getOutputValue();
}