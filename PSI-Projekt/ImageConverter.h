#pragma once
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

class ImageConverter
{
public:
	static std::vector<cv::Rect> detectLetters(cv::Mat img);
	static std::vector<cv::Rect> detectLetters2(cv::Mat img);
	static std::vector<std::vector<double>> prepareSamples(cv::Mat img);
	static std::vector<std::pair<cv::Rect,std::vector<double>>> prepareImg(cv::Mat img);
};