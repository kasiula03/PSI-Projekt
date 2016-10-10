#include "BaseTrainer.h"
#include <iostream>

BaseTrainer::BaseTrainer(vector<vector<double>> neurons, vector<double> targetVal, Neuron & neuron)
{
	double ni = 0.05;
	int inputCounter = neurons.back().size();
	for (int i = 0; i < neurons.size(); i++)
	{
		vector<double> inputs = neurons[i];
		//cout << "WEIGHTS ";
		for (int j = 0; j < inputs.size(); j++)
			cout << neuron.inputs[j].weight << " ";
		for (int j = 0; j < inputs.size(); j++)
			neuron.inputs[j].value = inputs[j];
	//	cout << endl;
		vector<double> weights;
		double output = neuron.calculateOutputValue();
		double delta  = targetVal[i] - output;
		for (int j = 0; j < inputs.size(); j++)
		{
			double newWeight = neuron.inputs[j].weight + (ni * delta * inputs[j]);
			weights.push_back(newWeight);
		}
		neuron.updateInputWeight(weights);
		cout << "\n input ";
		for (int j = 0; j < inputs.size(); j++)
		{
			cout << inputs[j] << " ";
		}
		cout << " output " << output << endl;
	
	}
	
	
}

void BaseTrainer::weigthTest(vector<Neuron> neurons, vector<double> targetVal)
{
	int inputCounter = neurons.back().inputs.size();
	for (int i = 0; i < neurons.size(); i++)
	{
		double output = neurons[i].calculateOutputValue();
		double delta = targetVal[i] - output;
		if (delta > 0.05)
			cout << "blad";
	}
}

