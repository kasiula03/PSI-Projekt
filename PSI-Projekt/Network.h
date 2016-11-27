#pragma once
#include "Neuron.h"

typedef vector<Neuron> Layer;

class Network
{
public:
	Network(vector<double> neuronsCount);
	void initializeInputs(vector<double> inputs, int numerOfLayer);
	void feedForward();
	void backPropagation(double targetsVal);
	void backPropagation(vector<double> targetsVal);
	vector<Layer> layers;
};