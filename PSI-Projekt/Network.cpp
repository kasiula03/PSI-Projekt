#include "Network.h"
#include <sstream>
#include <fstream>
#include <chrono>

Network::Network(vector<double> neuronsCount)
{
	for (int i = 1; i < neuronsCount.size(); ++i)
	{	
		Layer layer;
		for (int j = 0; j < neuronsCount[i]; ++j)
			layer.push_back(Neuron(neuronsCount[i-1]));
		layers.push_back(layer);
	}
	for (int i = 0; i < layers.size() - 1; ++i)
	{
		Layer & layer = layers[i];
		Layer & nextLayer = layers[i + 1];
		for (int j = 0; j < nextLayer.size(); ++j)
		{
			Neuron & nextLayerNeuron = nextLayer[j];
			for (int k = 0; k < nextLayerNeuron.inputs.size(); ++k)
				nextLayerNeuron.inputs[k].input = &layer[k]; // kazdemu inputowi warstwy nastepnej przypisujemy neuron z warszy poprzedniej
		}
	}
}

void Network::initializeInputs(vector<double> inputs, int numberOfLayer)
{
	Layer & firstLayer = layers.at(numberOfLayer);
	for (int i = 0; i < firstLayer.size(); ++i)
	{
		Neuron & neuron = firstLayer[i];
		for (int j = 0; j < neuron.inputs.size(); ++j)
			neuron.inputs[j].value = inputs[j];
	}
}
//seems ok
void Network::feedForward()
{
	for (unsigned i = 0; i < layers.size()-1; ++i) 
	{
		Layer& layer = layers[i];
		Layer& nextLayer = layers[i + 1];
		unsigned num_neuron = layer.size();
		for (unsigned n = 0; n < num_neuron; ++n)
			layer[n].feedForward();
		for (int j = 0; j < nextLayer.size(); ++j)
		{
			Neuron & neuron = nextLayer[j];
			for (int k = 0; k < neuron.inputs.size(); ++k)
				neuron.inputs[k].value = layer[k].getOutputValue();
			nextLayer[j].feedForward();
		}

	}
}

void Network::backPropagation(double targetsVal)
{
	Layer & outputLayer = layers.back();
	for (int i = 0; i < outputLayer.size(); ++i)
		outputLayer[i].calcOutputGradients(targetsVal);
	
	for (int i = layers.size() - 2; i >= 0; --i)
	{
		Layer & hiddenLayer = layers[i];
		Layer & nextLayer = layers[i + 1];
		double sum = 0.0;
		
		for (int j = 0; j < hiddenLayer.size(); ++j)
			hiddenLayer[j].calcHiddenGradients(nextLayer);
		
	}
	double ni = 0.3;
	for (int i = layers.size()-1; i > 0; --i)
	{
		Layer & currentLayer = layers[i];
		for (int j = 0; j < currentLayer.size(); ++j)
			currentLayer[j].updateInputWeight(ni);
	}
}

void Network::backPropagation(vector<double> targetsVal)
{
	Layer & outputLayer = layers.back();
	for (int i = 0; i < outputLayer.size(); ++i)
		outputLayer[i].calcOutputGradients(targetsVal[i]);

	for (int i = layers.size() - 2; i >= 0; --i)
	{
		Layer & hiddenLayer = layers[i];
		Layer & nextLayer = layers[i + 1];
		double sum = 0.0;
		for (int j = 0; j < hiddenLayer.size(); ++j)
			hiddenLayer[j].calcHiddenGradients(nextLayer);
	}
	double ni = 0.3;
	for (int i = layers.size()-1; i > 0; --i)
	{
		Layer & currentLayer = layers[i];
		for (int j = 0; j < currentLayer.size(); ++j)
			currentLayer[j].updateInputWeight(ni);
	}
}

void Network::saveWeights()
{
	auto t0 = std::chrono::system_clock::now();
	time_t time = std::chrono::system_clock::to_time_t(t0);
	stringstream ss;
	ss << time;
	string timeS = ss.str();
	string fileName = "weights_" + timeS + ".txt";
	int counter = 0;
	fstream file(fileName, ios::out);
	for (Layer layer : layers)
	{
		file << "layer_neurons: " << to_string(layer.size()) << endl;
		for (Neuron neuron : layer)
		{
			file << "inputs: " << neuron.inputs.size() << endl;
			for (Connection connection : neuron.inputs)
			{
				file << to_string(connection.weight) << endl;
			}
		}
	}

	file.close();
}

void Network::loadNetwork(string fileName)
{
	vector<double> layersSize;
	vector<vector<double>> weights;
	fstream file(fileName, ios::in);
	int layerNumber = 0;
	int layerSize = 0;
	int counter = 0;
	int inputSize = 0;
	string line;
	file >> line;
	while(!file.eof())
	{
		
		if (line == "layer_neurons:")
		{
			counter = 0;
			string next;
			file >> next;
			layerSize = atof(next.c_str());
			layerNumber++;
			file >> line;
		}
		
		if (line == "inputs:")
		{
			file >> line;
			inputSize = atoi(line.c_str());
			string value;
			for (int i = 0; i < inputSize; i++)
			{
				file >> value;
				double val = atof(value.c_str());
				layers[layerNumber - 1][counter].inputs[i].weight = val;
			}
			file >> line;
			counter++;
		}
	}
	file.close();
}