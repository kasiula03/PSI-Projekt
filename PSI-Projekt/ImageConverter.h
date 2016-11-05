#pragma once
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

class ImageConverter
{
public:
	std::vector<cv::Rect> detectLetters(cv::Mat img);
	std::vector<cv::Rect> detectLetters2(cv::Mat img);
	int prepareSamples(cv::Mat img);
};