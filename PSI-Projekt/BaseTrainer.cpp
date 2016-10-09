#include "BaseTrainer.h"
#include <iostream>

BaseTrainer::BaseTrainer(vector<Neuron> & neurons, vector<double> targetVal)
{
	int inputCounter = neurons.back().inputs.size();
	for (int i = 0; i < neurons.size(); i++)
	{
		double delta = 1.0;
		while (delta > 0.05)
		{
			vector<double> weights;
			double output = neurons[i].calculateOutputValue();
			delta = targetVal[i] - output;
			for (int j = 0; j < inputCounter; j++)
			{
				weights.push_back(neurons[i].inputs[j].weight + (delta * targetVal[i]));
			}
			neurons[i].updateInputWeight(weights);

		}
		for (int j = 0; j < inputCounter; j++)
		{
			cout << neurons[i].inputs[j].weight << " ";
		}
		cout << endl;
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

