#include "Neuron.h"
#include <iostream>
#include <cmath>

using namespace std;

Neuron::Neuron(unsigned inputSize)
{
	for (int i = 0; i < inputSize; i++)
	{
		inputs.push_back(Connection());
		inputs.back().weight = randomWeight();

	}
	outputValue = calculateOutputValue();
	
}



void Neuron::showNeuron()
{
	cout << "Weights: ";
	for (int i = 0; i < inputs.size(); ++i)
	{
		cout << inputs[i].weight << " ";
	}
	cout << "\n Inputs: ";
	for (int i = 0; i < inputs.size(); ++i)
	{
		cout << inputs[i].value << " ";
	}
	cout << endl;
	cout << "Output " << getOutputValue() << endl;
}

void Neuron::feedForward()
{
	double sum = 0.0;
	for (int i = 0; i < inputs.size(); ++i)
	{
		sum += inputs[i].value * inputs[i].weight;
	}
	outputValue = Neuron::activatorFun(sum);
}

void Neuron::feedForward(const vector<double> &prevLayer)
{
	double sum = 0.0;
	for (int i = 0; i < prevLayer.size(); ++i)
	{
		sum += prevLayer[i] * inputs[i].weight;
	}
	outputValue = Neuron::activatorFun(sum);
}

double Neuron::activatorFun(double x)
{
	if (x >= 0.5f) return 1;
	else return 0;

	//double mian = 1 + exp(-x);
	//return (1 / mian);
	//return tanh(x);
}

double Neuron::derivativeActivatorFun(double x)
{
	return Neuron::activatorFun(x)*(1 - Neuron::activatorFun(x));
	//return (1.0 - x*x);
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

void Neuron::calcOutputGradients(double targetVal)
{
	double delta = targetVal - outputValue;
	//gradient = delta * Neuron::derivativeActivatorFun(outputValue);
	gradient = delta;
}

void Neuron::calcHiddenGradients(const vector<Neuron> &nextLayer)
{
	double sum = 0.0;
	for (int i = 0; i < nextLayer.size(); i++)
	{
		for (int j = 0; j < nextLayer[i].inputs.size(); ++j)
		{
			if (nextLayer[i].inputs[j].input == this)
			{
				sum += (nextLayer[i].inputs[j].weight * nextLayer[i].gradient);
			}
		}
		
		
	}
	//gradient = sum * Neuron::derivativeActivatorFun(outputValue);
	gradient = sum;
}

void Neuron::updateInputWeight(double q)
{
	double sum = 0.0;
	for (int i = 0; i < inputs.size(); ++i)
	{
		sum += (inputs[i].value * inputs[i].weight);
	}
	for (int i = 0; i < inputs.size(); ++i)
	{
		double deltaWeight = q * gradient * Neuron::derivativeActivatorFun(sum) * inputs[i].value;
		inputs[i].weight += deltaWeight;
	}
}