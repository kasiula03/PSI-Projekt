#include "Neuron.h"
#include <cmath>

Neuron::Neuron(unsigned inputSize)
{
	for (int i = 0; i < inputSize; i++)
	{
		inputs.push_back(Connection());
		inputs.back().weight = randomWeight();
	}
	outputValue = calculateOutputValue();
}

double Neuron::activatorFun(double x)
{
	//return tanh(x);
	if (x > 0.5f) return 1;
	else return 0;
}

double Neuron::derivativeActivatorFun(double x)
{
	return (1.0 - x * x);
}

void Neuron::updateInputWeight(vector <double> weight)
{
	for (int i = 0; i < weight.size(); i++)
	{
		inputs[i].weight = weight[i];
	}
}

double Neuron::randomWeight()
{
	return (double)rand() / RAND_MAX;
}

double Neuron::calculateOutputValue()
{
	double sum = 0.0;
	for (int i = 0; i < inputs.size(); i++)
	{
		Connection connection = inputs[i];
		sum += (connection.weight * connection.value);
	}
	this->outputValue = activatorFun(sum);
	return outputValue;
}