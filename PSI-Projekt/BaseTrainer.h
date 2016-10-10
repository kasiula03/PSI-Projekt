#pragma once
#include "Neuron.h"

typedef vector<Neuron> layer;

class BaseTrainer
{
public:
	BaseTrainer(vector<vector<double>>, vector<double> targetVal, Neuron & neuron);
	void weigthTest(vector<Neuron> neurons, vector<double> targetVal);
private:
	vector<layer> net;
	double ni = 0.05;
};