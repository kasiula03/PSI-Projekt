#pragma once
#include "Neuron.h"

typedef vector<Neuron> Layer;

class Network
{
public:
	Network(vector<double> neuronsCount);
	void loadNetwork(string fileName);
	void initializeInputs(vector<double> inputs, int numerOfLayer);
	void feedForward();
	void saveWeights();
	void backPropagation(double targetsVal);
	void backPropagation(vector<double> targetsVal);
	vector<Layer> layers;
};