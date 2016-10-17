#include "BaseTrainer.h"
#include <iostream>
#include <fstream>

BaseTrainer::BaseTrainer(vector<vector<double>> neurons, vector<double> targetVal, Neuron & neuron)
{
	double ni = 0.05;
	int inputCounter = neurons.back().size();

	for (int i = 0; i < neurons.size(); i++)
	{
		showResult(neuron);
		vector<double> inputs = neurons[i];
		for (int j = 0; j < inputs.size(); j++)
			neuron.inputs[j].value = inputs[j];
		vector<double> weights;
		double output = neuron.calculateOutputValue();
		double delta  = targetVal[i] - output;

		for (int j = 0; j < inputs.size(); j++)
		{
			double newWeight = neuron.inputs[j].weight + (ni * delta * inputs[j]);
			weights.push_back(newWeight);
		}
		neuron.updateInputWeight(weights);
	}
	
}

BaseTrainer::BaseTrainer(vector<vector<double >> inputs, vector<double> targetVal, Network & network)
{
	double ni = 0.05;
	int inputCounter = inputs.back().size();
	vector<double> deltas;
	for (int i = 0; i < inputs.size(); i++)
	{
		vector<double> currentInputs = inputs[i];
		network.initializeInputs(currentInputs, 0);
		network.feedForward();
		double delta = targetVal[i] - network.layers.back()[0].getOutputValue();
		deltas.push_back(delta);
		cout << delta << endl;
		//showResult(network);
		network.backPropagation(targetVal);
	}
	saveDelatasToCSV(deltas);
}

void BaseTrainer::saveDelatasToCSV(vector<double> deltas)
{
	ofstream fout("deltas.csv");
	for (int i = 0; i < deltas.size(); ++i)
	{
		fout << deltas[i] << "\n";
	}
	fout.close();
}

void BaseTrainer::showResult(const Neuron & neuron)
{
	int inputCount = neuron.inputs.size();
	cout << "\nWeights: ";
	for (int i = 0; i < inputCount; i++)
	{
		cout << neuron.inputs[i].weight << " ";
	}
	cout << "\n Inputs: ";
	for (int i = 0; i < inputCount; i++)
	{
		cout << neuron.inputs[i].value << " ";
	}
	cout << "\n Output: " << neuron.getOutputValue();
}

void BaseTrainer::showResult(Network network)
{
	for (int i = 0; i < network.layers.size(); i++)
	{
		cout << "Layer " << i << endl;
		for (int j = 0; j < network.layers[i].size(); j++)
		{
			network.layers[i][j].showNeuron();
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

