#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <chrono>
#include <opencv2/core/core.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "ImageConverter.h"
#include "BaseTrainer.h"
#include "MulipleOutputTrainer.h"
#include "ActivatorFunctions.h"
#include "Network.h"

using namespace cv;
using namespace std;

void xor();
vector<vector<double>> parseOutputVal();
vector<double> getImgAsFloats(Mat image);

int main()
{
	srand(time(NULL));
	Mat img1 = cv::imread("test_letter.png");
	Mat img2 = cv::imread("test_letter2.png");
	vector<double> im = getImgAsFloats(img1);
	vector<double> im2 = getImgAsFloats(img2);
	int m = 0;
	for (int j = 0; j < 30; ++j)
	{
		for (int k = 0; k < 20; ++k)
		{
			if (im[m] > 0)
				im[m] = 1;
			else
				im[m] = -1;
			cout << im[m];
			m++;
		}
		cout << endl;
	}
	m = 0;
	for (int j = 0; j < 30; ++j)
	{
		for (int k = 0; k < 20; ++k)
		{
			if (im2[m] > 0)
				im2[m] = 1;
			else
				im2[m] = -1;
			cout << im2[m];
			m++;
		}
		cout << endl;
	}
	Neuron::activator = ActivatorFunctions::sigmoid;
	Neuron::derivativeActivator = ActivatorFunctions::derivativeSigmoid;
	ImageConverter converter;
	Mat traingImg = imread("training_chars.png");
	vector<vector<float>> map = converter.prepareSamples(traingImg);
	
	for (int i = 0; i < map.size(); ++i)
	{
		m = 0;
		for (int j = 0; j < 30; ++j)
		{
			for (int k = 0; k < 20; ++k)
			{
				cout << map[i][m];
				m++;
			}
			cout << endl;
		}
		cout << endl;
	}
	vector<vector<double>> normalized;
	for (int i = 0; i < map.size(); ++i)
	{
		vector<double> each;
		for (int j = 0; j < map[i].size(); ++j)
		{
			if (map[i][j] > 0)
				each.push_back(1);
			else
				each.push_back(-1);
		}
		normalized.push_back(each);
	}
	vector<vector<double>> targets = parseOutputVal();
	double inputsSize = normalized[0].size();
	double outputsSize = targets[0].size();
	vector<double> vec{inputsSize, sqrt(inputsSize), outputsSize };
	Network network(vec);
	MultipleOutputTrainer multipleTrainer(normalized, targets, network);
	

	network.initializeInputs(im, 0);
	network.feedForward();
	Layer & outputLayer = network.layers.back();
	cout << "\n\n\nResults\n\n";
	for (int i = 0; i < outputLayer.size(); ++i)
		cout << i << "\t" << outputLayer[i].getOutputValue() << endl;
	

	network.initializeInputs(im2, 0);
	network.feedForward();
	outputLayer = network.layers.back();
	cout << "\n\n\nResults\n\n";
	for (int i = 0; i < outputLayer.size(); ++i)
		cout << i << "\t" << outputLayer[i].getOutputValue() << endl;
	/*cv::Mat img1 = cv::imread("test.jpg");
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
	//cv::imwrite("ConvertedImages/ttt2.jpg", part);*/

	
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

vector<double> getImgAsFloats(Mat image)
{
	Mat imgTrainingNumbers, imgGrayscale, imgBlurred, imgThresh, imgThreshCopy;
	imgTrainingNumbers = image;
	vector<vector<Point>> ptContours; // declare contours vector
	vector<Vec4i> v4iHierarchy; // declare contours hierarchy

	Mat matTrainingImagesAsFlattenedFloats;


	if (imgTrainingNumbers.empty())
	{
		cout << "Error: image not exist! \n";
		
	}
	cvtColor(imgTrainingNumbers, imgGrayscale, CV_BGR2GRAY); // convert to grayscale
	GaussianBlur(imgGrayscale, imgBlurred, Size(5, 5), 0);
	adaptiveThreshold(imgBlurred,
		imgThresh,
		255,							// make pixels that pass the threshold full white
		ADAPTIVE_THRESH_GAUSSIAN_C,
		THRESH_BINARY_INV,				// invert so foreground will be white, background will be black
		11,								// size of a pixel neighborhood used to calculate threshold value
		2
		);
	imgThreshCopy = imgThresh.clone();
	findContours(imgThreshCopy,
		ptContours,
		v4iHierarchy,
		RETR_EXTERNAL,
		CHAIN_APPROX_SIMPLE);
	//imwrite("Thresh1.jpg", imgThresh);
	//imwrite("Thresh2.jpg", imgThreshCopy);
	Mat binary;
	vector<vector<double>> map;
	vector<double> vec;
	for (int i = 0; i < ptContours.size(); ++i)
	{
		if (contourArea(ptContours[i]) > 100)
		{
			Rect boundRect = boundingRect(ptContours[i]);
			rectangle(imgTrainingNumbers, boundRect, Scalar(0, 0, 255), 2);
			Mat matPart = imgThresh(boundRect); // part of image
			Mat matPartResized;
			resize(matPart, matPartResized, Size(20, 30));
			imwrite("resize.jpg", matPartResized);
			//Mat binary(matPartResized.size(), matPartResized.type());
			threshold(matPartResized, binary, 100, 255, THRESH_BINARY);
			Mat matImageFloat;
			matPartResized.convertTo(matImageFloat, CV_32FC3, 1 / 255.0);
			Mat matImageFlattenedFloat = matImageFloat.reshape(1, 1);
			matTrainingImagesAsFlattenedFloats.push_back(matImageFlattenedFloat);
			matImageFlattenedFloat.row(0).copyTo(vec);
			std::vector<double> array;
			if (matImageFlattenedFloat.isContinuous()) {
				array.assign((double*)matImageFlattenedFloat.datastart, (double*)matImageFlattenedFloat.dataend);
			}
			else {
				for (int i = 0; i < matImageFlattenedFloat.rows; ++i) {
					array.insert(array.end(), (double*)matImageFlattenedFloat.ptr<uchar>(i), (double*)matImageFlattenedFloat.ptr<uchar>(i) + matImageFlattenedFloat.cols);
				}
			}
			map.push_back(array);
		}
	}
	cv::FileStorage fsTrainingImages("images2.xml", cv::FileStorage::WRITE);         // open the training images file

	if (fsTrainingImages.isOpened() == false) {                                                 // if the file was not opened successfully
		std::cout << "error, unable to open training images file, exiting program\n\n";         // show error message
		                                                                             // and exit program
	}
	
	fsTrainingImages << "images" << matTrainingImagesAsFlattenedFloats;         // write training images into images section of images file
	fsTrainingImages.release();
	
	return vec;
}

vector<vector<double>> parseOutputVal()
{
	vector<int> intValidChars = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' };

	vector<vector<double>> outputs;
	for (int k = 0; k < 5; ++k)
	{
		for (int i = 0; i < intValidChars.size(); ++i)
		{
			vector<double> each;
			for (int j = 0; j < intValidChars.size(); ++j)
			{
				if (i == j)
					each.push_back(1);
				else
					each.push_back(-1);
			}
			outputs.push_back(each);
		}
	}
	
	return outputs;
}

void xor()
{
	Neuron::activator = ActivatorFunctions::sigmoid;
	Neuron::derivativeActivator = ActivatorFunctions::derivativeSigmoid;
	vector<double> vec{ 2, 4, 1 };
	Network network(vec);

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