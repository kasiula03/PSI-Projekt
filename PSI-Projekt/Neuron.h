#pragma once
#include <vector>
#include "Connection.h"

using namespace std;


class Neuron
{
public:
	Neuron(unsigned inputSize);
	vector <Connection> inputs;

	inline void setOutputValue(double x) { outputValue = x; }
	inline double getOutputValue() const { return outputValue; }

	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const vector<Neuron> &nextLayer);
	void updateInputWeight(vector <double> weight);
	void updateInputWeight(double q);
	void showNeuron();
	double calculateOutputValue();
	static double activatorFun(double x);
	static double derivativeActivatorFun(double x);
	double gradient;

private:
	static double randomWeight();
	double outputValue;
	
};
