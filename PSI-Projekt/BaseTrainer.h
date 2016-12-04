#pragma once
#include "Network.h"

typedef vector<Neuron> layer;

class BaseTrainer
{
public:
	BaseTrainer(vector<vector<double>>, vector<double> targetVal, Neuron & neuron);
	BaseTrainer(vector<vector<double >> inputs, vector<double> targetVal, Network & network);
	void testNetwork(vector<vector<double>>, vector<double>, Network &);
	void showResult(const Neuron & neuron);
	void showResult(Network network);
	static void saveDelatasToCSV(vector<double>);
private:
	vector<layer> net;
	vector<double> errors;

	double ni = 0.05;
};