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
	void calcHiddenGradients(vector<Neuron> &nextLayer);
	void updateHebbInputWeight(double q);
	void updateInputWeight(vector <double> weight);
	void updateInputWeight(double q);
	void showNeuron();
	void feedForward();
	double calculateOutputValue();

	double gradient;
	double outputValue;

	// activator function
	static double(*activator)(double x);
	static double(*derivativeActivator)(double x);
	static double activatorFun(double x);
	static double derivativeActivatorFun(double x);

private:
	static double randomWeight();

	
};
