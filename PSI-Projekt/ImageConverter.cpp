#include "ImageConverter.h"
#include <iostream>

using namespace cv;
using namespace std;

vector<Rect> ImageConverter::detectLetters(Mat img)
{

	vector<Rect> boundRect;
	Mat img_gray, img_sobel, img_threshold, blur_image, element;
	cvtColor(img, img_gray, CV_BGR2GRAY);
	Sobel(img_gray, img_sobel, CV_8U, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	threshold(img_sobel, img_threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
	element = getStructuringElement(MORPH_ELLIPSE, Size(20, 6));
	morphologyEx(img_threshold, img_threshold, CV_MOP_CLOSE, element); //Does the trick
	vector<vector<Point> > contours;
	findContours(img_threshold, contours, 0, 1);
	vector<std::vector<Point> > contours_poly(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		if (contours[i].size()>100)
		{
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			Rect appRect(boundingRect(Mat(contours_poly[i])));
			if (appRect.width > appRect.height)
				boundRect.push_back(appRect);
		}
	}
	
	

	return boundRect;
}

vector<vector<float>> ImageConverter::prepareSamples(Mat img)
{
	Mat imgTrainingNumbers, imgGrayscale, imgBlurred, imgThresh, imgThreshCopy; 
	imgTrainingNumbers = img;
	vector<vector<Point>> ptContours; // declare contours vector
	vector<Vec4i> v4iHierarchy; // declare contours hierarchy

	Mat matTrainingImagesAsFlattenedFloats;
	

	if (imgTrainingNumbers.empty())
	{
		cout << "Error: image not exist! \n";
		return vector<vector<float>>();
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
	vector<vector<float>> map;
	for (int i = 0; i < ptContours.size(); ++i)
	{
		if (contourArea(ptContours[i]) > 100)
		{
			Rect boundRect = boundingRect(ptContours[i]);
			rectangle(imgTrainingNumbers, boundRect, Scalar(0, 0, 255), 2);
			Mat matPart = imgThresh(boundRect); // part of image
			Mat matPartResized;
			Mat binary;
			resize(matPart, matPartResized, Size(20,30));
			threshold(matPartResized, binary, 100, 255, THRESH_BINARY);
			Mat matImageFloat = binary;
			matPartResized.convertTo(matImageFloat, CV_32FC3);
			Mat matImageFlattenedFloat = matImageFloat.reshape(1, 1);
			matTrainingImagesAsFlattenedFloats.push_back(binary);
			std::vector<float> array;
			if (binary.isContinuous()) {
				array.assign((float*)matImageFlattenedFloat.datastart, (float*)matImageFlattenedFloat.dataend);
			}
			else {
				for (int i = 0; i < matImageFlattenedFloat.rows; ++i) {
					array.insert(array.end(), (float*)matImageFlattenedFloat.ptr<uchar>(i), (float*)matImageFlattenedFloat.ptr<uchar>(i) + matImageFlattenedFloat.cols);
				}
			}
			vector<float> vec;
			matImageFlattenedFloat.row(0).copyTo(vec);
			int m = 0;
			for (int j = 0; j < 30; ++j)
			{
				for (int k = 0; k < 20; ++k)
				{
					if (vec[m] > 0)
						vec[m] = 1;
					else
						vec[m] = -1;
					m++;
				}
			}
			map.push_back(vec);
		}
	}
	cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::WRITE);         // open the training images file
	
	if (fsTrainingImages.isOpened() == false) {                                                 // if the file was not opened successfully
		std::cout << "error, unable to open training images file, exiting program\n\n";         // show error message
		return vector<vector<float>>();                                                                              // and exit program
	}

	fsTrainingImages << "images" << matTrainingImagesAsFlattenedFloats;         // write training images into images section of images file
	fsTrainingImages.release();

	
	return map;
}

vector<Rect> ImageConverter::detectLetters2(Mat img)
{

	//Apply blur to smooth edges and use adapative thresholding
	cv::Size size(3, 3);
	cv::GaussianBlur(img, img, size, 0);
	adaptiveThreshold(img, img, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 75, 10);
	cv::bitwise_not(img, img);

	cv::Mat img2 = img.clone();


	std::vector<cv::Point> points;
	cv::Mat_<uchar>::iterator it = img.begin<uchar>();
	cv::Mat_<uchar>::iterator end = img.end<uchar>();
	for (; it != end; ++it)
		if (*it)
			points.push_back(it.pos());

	cv::RotatedRect box = cv::minAreaRect(cv::Mat(points));

	double angle = box.angle;
	if (angle < -45.)
		angle += 90.;

	cv::Point2f vertices[4];
	box.points(vertices);
	for (int i = 0; i < 4; ++i)
		cv::line(img, vertices[i], vertices[(i + 1) % 4], cv::Scalar(255, 0, 0), 1, CV_AA);



	cv::Mat rot_mat = cv::getRotationMatrix2D(box.center, angle, 1);

	cv::Mat rotated;
	cv::warpAffine(img2, rotated, rot_mat, img.size(), cv::INTER_CUBIC);



	cv::Size box_size = box.size;
	if (box.angle < -45.)
		std::swap(box_size.width, box_size.height);
	cv::Mat cropped;

	cv::getRectSubPix(rotated, box_size, box.center, cropped);
	imwrite("example5.jpg", cropped);

	Mat cropped2 = cropped.clone();
	cvtColor(cropped2, cropped2, CV_GRAY2RGB);

	Mat cropped3 = cropped.clone();
	cvtColor(cropped3, cropped3, CV_GRAY2RGB);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// Find contours
	cv::findContours(cropped, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_KCOS, Point(0, 0));

	/// Approximate contours to polygons + get bounding rects and circles
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f>center(contours.size());
	vector<float>radius(contours.size());


	//Get poly contours
	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
	}


	//Get only important contours, merge contours that are within another
	vector<vector<Point> > validContours;
	for (int i = 0;i<contours_poly.size();i++) {

		Rect r = boundingRect(Mat(contours_poly[i]));
		if (r.area()<100)continue;
		bool inside = false;
		for (int j = 0;j<contours_poly.size();j++) {
			if (j == i)continue;

			Rect r2 = boundingRect(Mat(contours_poly[j]));
			if (r2.area()<100 || r2.area()<r.area())continue;
			if (r.x>r2.x&&r.x + r.width<r2.x + r2.width&&
				r.y>r2.y&&r.y + r.height<r2.y + r2.height) {

				inside = true;
			}
		}
		if (inside)continue;
		validContours.push_back(contours_poly[i]);
	}

	//Get bounding rects
	for (int i = 0;i<validContours.size();i++) {
		boundRect[i] = boundingRect(Mat(validContours[i]));
	}

	return boundRect;
}