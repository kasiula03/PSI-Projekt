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
	for (int i = 0; i < layers.size() - 1; ++i)
	{
		Layer & layer = layers[i];
		for (int j = 0; j < layers[i + 1].size(); ++j)
		{
			for (int k = 0; k < layers[i + 1][j].inputs.size(); ++k)
			{
				layers[i + 1][j].inputs[k].input = &layer[k];
			}
			
		}
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
//seems ok
void Network::feedForward()
{
	for (unsigned i = 0; i < layers.size()-1; ++i) 
	{
		Layer& layer = layers[i];
		Layer& nextLayer = layers[i + 1];
		unsigned num_neuron = (layer.size());
		for (unsigned n = 0; n < num_neuron; ++n)
			layer[n].feedForward();
		for (int j = 0; j < nextLayer.size(); ++j)
		{
			Neuron & neuron = nextLayer[j];
			for (int k = 0; k < nextLayer[j].inputs.size(); ++k)
			{
				neuron.inputs[k].value = layer[k].getOutputValue();
			}
			nextLayer[j].feedForward();
		}

	}
}

void Network::backPropagation(double targetsVal)
{
	Layer & outputLayer = layers.back();
	for (int i = 0; i < outputLayer.size(); ++i)
	{
		double delta = targetsVal - outputLayer[i].getOutputValue();
		outputLayer[i].calcOutputGradients(targetsVal);
	}

	for (int i = layers.size() - 2; i >= 0; --i)
	{
		Layer & hiddenLayer = layers[i];
		Layer & nextLayer = layers[i + 1];
		double sum = 0.0;
		
		for (int j = 0; j < hiddenLayer.size(); ++j)
		{
			hiddenLayer[j].calcHiddenGradients(nextLayer);
		}
			
	}
	double ni = 0.3;
	for (int i = 0; i < layers.size(); ++i)
	{
		Layer & currentLayer = layers[i];
		for (int j = 0; j < currentLayer.size(); ++j)
		{
			currentLayer[j].updateInputWeight(ni);
		}
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
		double sum = 0.0;
		for (int j = 0; j < hiddenLayer.size(); ++j)
			hiddenLayer[j].calcHiddenGradients(nextLayer);
		

	}
	double ni = 0.3;
	for (int i = 0; i < layers.size(); ++i)
	{
		Layer & currentLayer = layers[i];
		for (int j = 0; j < currentLayer.size(); ++j)
			currentLayer[j].updateInputWeight(ni);
	}
}