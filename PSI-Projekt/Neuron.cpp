#include "Neuron.h"
#include <iostream>


using namespace std;

double(*Neuron::activator)(double x);
double(*Neuron::derivativeActivator)(double x);

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
	calculateOutputValue();
}

double Neuron::activatorFun(double x)
{
	return activator(x);
}

double Neuron::derivativeActivatorFun(double x)
{
	return derivativeActivator(x);
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
	return (double(rand() % 200) - 100) / 100;
}

double Neuron::calculateOutputValue()
{
	double sum = 0.0;
	for (int i = 0; i < inputs.size(); i++)
	{
		Connection & connection = inputs[i];
		sum += (connection.weight * connection.value);
		
	}
	this->outputValue = activator(sum);
	return outputValue;
}

void Neuron::calcOutputGradients(double targetVal)
{
	double delta = targetVal - outputValue;
	gradient = delta;
}

void Neuron::calcHiddenGradients(vector<Neuron> &nextLayer)
{
	double sum = 0.0;
	for (int i = 0; i < nextLayer.size(); i++)
	{
		Neuron & neuron = nextLayer[i];
		for (int j = 0; j < neuron.inputs.size(); ++j)
		{
			Connection & con = neuron.inputs[j];
			if (con.input == this)
			{
				sum += (con.weight * neuron.gradient);
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
		double deltaWeight = q * gradient * derivativeActivator(sum) * inputs[i].value;
		inputs[i].weight += deltaWeight;
	}
}