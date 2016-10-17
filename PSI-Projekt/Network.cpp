#include "Network.h"

Network::Network(vector<double> neuronsCount)
{
	for (int i = 1; i < neuronsCount.size(); ++i)
	{	
		Layer layer;
		for (int j = 0; j < neuronsCount[i]; ++j)
		{
			layer.push_back(Neuron(neuronsCount[i-1]));
		}
		layers.push_back(layer);
	}
}

void Network::initializeInputs(vector<double> inputs, int numberOfLayer)
{
	Layer & firstLayer = layers.at(numberOfLayer);
	for (int i = 0; i < firstLayer.size(); ++i)
	{
		for (int j = 0; j < firstLayer[i].inputs.size(); ++j)
			firstLayer[i].inputs[j].value = inputs[j];
	}
}

void Network::feedForward()
{
	for (int i = 0; i < layers.size(); i++)
	{
		vector<double> nextInputs;
		for (int j = 0; j < layers[i].size(); ++j)
		{
			layers[i][j].calculateOutputValue();
			nextInputs.push_back(layers[i][j].getOutputValue());
		}
		if(i + 1 < layers.size())
			initializeInputs(nextInputs, i + 1);
	}
}

void Network::backPropagation(vector<double> targetsVal)
{
	Layer & outputLayer = layers.back();
	for (int i = 0; i < outputLayer.size(); ++i)
	{
		double delta = targetsVal[i] - outputLayer[i].getOutputValue();
		outputLayer[i].calcOutputGradients(targetsVal[i]);
	}

	for (int i = layers.size() - 2; i >= 0; --i)
	{
		Layer & hiddenLayer = layers[i];
		Layer & nextLayer = layers[i + 1];
		for (int j = 0; j < hiddenLayer.size(); ++j)
			hiddenLayer[j].calcHiddenGradients(nextLayer);
	}

	for (int i = layers.size() - 1; i >= 0; --i)
	{
		Layer & currentLayer = layers[i];
		for (int j = 0; j < currentLayer.size(); ++j)
		{
			currentLayer[j].updateInputWeight(0.2);
		}
			//currentLayer[j].updateInputWeight(targetsVal[j], 0.05);
	}
}