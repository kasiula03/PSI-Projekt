#include "ColorAnalize.h"
#include <iostream>

void ColorAnalize::analizeColor(Vec4b pixel, Neuron & neuron)
{
	for (int i = 0; i < 4; i++)
	{
		neuron.inputs[i].value = pixel.val[i];
	}
	std::cout << neuron.calculateOutputValue() << std::endl;
}