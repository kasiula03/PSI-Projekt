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
	inline double getOutputValue() { return outputValue; }

	double calculateOutputValue();
	void updateInputWeight(vector <double> weight);
	static double activatorFun(double x);
	static double derivativeActivatorFun(double x);
	double gradient;

private:
	static double randomWeight();
	double outputValue;
	
};