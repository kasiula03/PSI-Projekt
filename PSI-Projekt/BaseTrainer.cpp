#include "BaseTrainer.h"
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

BaseTrainer::BaseTrainer(vector<vector<double>> neurons, vector<double> targetVal, Neuron & neuron)
{
	double ni = 0.05;
	int inputCounter = neurons.back().size();
	double delta = 1;
	while (abs(0.5 * delta * delta) > 0.0005)
	{
		for (int i = 0; i < neurons.size(); i++)
		{
			vector<double> inputs = neurons[i];
			for (int j = 0; j < inputs.size(); j++)
				neuron.inputs[j].value = inputs[j];
			showResult(neuron);
			vector<double> weights;
			double output = neuron.calculateOutputValue();
			delta = targetVal[i] - output;

			for (int j = 0; j < inputs.size(); j++)
			{
				double newWeight = neuron.inputs[j].weight + (ni * delta * inputs[j]);
				weights.push_back(newWeight);
			}
			neuron.updateInputWeight(weights);
		}
	}
	
}

BaseTrainer::BaseTrainer(vector<vector<double >> inputs, vector<double> targetVal, Network & network)
{
	double ni = 0.05;
	int inputCounter = inputs.back().size();
	vector<double> deltas;
	double delta = 1;
	int counter = 0;
	while (abs(0.5 * delta * delta) > 0.005)
	{
		for (int i = 0; i < inputs.size(); i++)
		{
			vector<double> currentInputs = inputs[i];
			network.initializeInputs(currentInputs, 0);
			network.feedForward();
			delta = targetVal[i] - network.layers.back()[0].getOutputValue();
			deltas.push_back(delta);
			cout << "Blad" << 0.5 * delta * delta << endl;
			//showResult(network);
			network.backPropagation(targetVal[i]);
		}
		if (counter % 10 == 0)
		{
			//testNetwork(inputs, targetVal, network);
			errors.push_back(abs(0.5 * delta * delta));
		}
		counter++;
	}
	saveDelatasToCSV(errors);
}

void BaseTrainer::testNetwork(vector<vector<double>> inputs, vector<double> target, Network & network)
{
	for (int i = 0; i < inputs.size(); ++i)
	{
		vector<double> input = inputs[i];
		network.initializeInputs(input, 0);
		network.feedForward();
		double delta = target[i] - network.layers.back().back().getOutputValue();
	//	errors.push_back(delta);
	}
	
}

void BaseTrainer::saveDelatasToCSV(vector<double> deltas)
{
	ofstream fout("deltas.csv");
	for (int i = 0; i < deltas.size(); ++i)
	{
		string s = to_string(deltas[i]);
		replace(s.begin(), s.end(), '.', ',');
		fout << s << "\n";
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


