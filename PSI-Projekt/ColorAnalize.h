#pragma once
#include<opencv2/core/core.hpp>

#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include"Neuron.h"

using namespace cv;

class ColorAnalize
{
public:
	void analizeColor(Vec4b pixel, Neuron & neuron);
};