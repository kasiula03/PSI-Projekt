#pragma once
#include "Network.h"

typedef vector<Neuron> layer;

class BaseTrainer
{
public:
	BaseTrainer(vector<vector<double>>, vector<double> targetVal, Neuron & neuron);
	BaseTrainer(vector<vector<double >> inputs, vector<double> targetVal, Network & network);
	void weigthTest(vector<Neuron> neurons, vector<double> targetVal);
	void showResult(const Neuron & neuron);
	void showResult(Network network);
private:
	vector<layer> net;
	static void saveDelatasToCSV(vector<double>);
	double ni = 0.05;
};