#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>

#include<opencv2/core/core.hpp>

#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include "BaseTrainer.h"
#include "Network.h"

using namespace cv;
using namespace std;

int main()
{
	srand(time(NULL));
	vector<double> vec{ 2,2,1 };
	Network network(vec);
	//network.initializeInputs(input, 0);
	//network.feedForward();

	//xor
	vector<vector<double>> neurons;
	vector<double> targetVal;
	for (int i = 20000; i >= 0; --i)
	{
		int n1 = (int)(2.0 * rand() / double(RAND_MAX));
		int n2 = (int)(2.0 * rand() / double(RAND_MAX));
		int t;
		if ((n1 == 0 && n2 == 0) || (n1 == 1 && n2 == 1))
			t = 0;
		else
			t = 1;

		vector<double> neuron;
		neuron.push_back(n1);
		neuron.push_back(n2);
		neurons.push_back(neuron);
		targetVal.push_back(t);
	}
	BaseTrainer trainer(neurons, targetVal, network);
	vector<double> inp = { 0,1 };
	network.initializeInputs(inp, 0);
	network.feedForward();
	cout << "\nTest : " << inp[0] << " " << inp[1] << endl;
	cout << "\n " << network.layers.back().back().getOutputValue();

	inp = { 1,0 };
	network.initializeInputs(inp, 0);
	network.feedForward();
	cout << "\nTest : " << inp[0] << " " << inp[1] << endl;
	cout << "\n " << network.layers.back().back().getOutputValue();

	inp = { 0,0 };
	network.initializeInputs(inp, 0);
	network.feedForward();
	cout << "\nTest : " << inp[0] << " " << inp[1] << endl;
	cout << "\n " << network.layers.back().back().getOutputValue();

	inp = { 1,1 };
	network.initializeInputs(inp, 0);
	network.feedForward();
	cout << "\nTest : " << inp[0] << " " << inp[1] << endl;
	cout << "\n " << network.layers.back().back().getOutputValue();
/*
	vector<vector<double>> ANDneurons;
	vector<double> ANDtargetVal;
	Neuron neuronAND(2);
	for (int i = 2000; i >= 0; --i)
	{
		int n1 = (int)(2.0 * rand() / double(RAND_MAX));
		int n2 = (int)(2.0 * rand() / double(RAND_MAX));

		int t = n1 & n2;
		vector<double> neuron;
		neuron.push_back(n1);
		neuron.push_back(n2);
		ANDneurons.push_back(neuron);
		ANDtargetVal.push_back(t);
	}

	BaseTrainer trainerAND(ANDneurons, ANDtargetVal, neuronAND);
	neuronAND.inputs[0].value = 0;
	neuronAND.inputs[1].value = 0;
	cout << "\n" << neuronAND.calculateOutputValue();
	neuronAND.inputs[0].value = 1;
	neuronAND.inputs[1].value = 0;
	cout << "\n" << neuronAND.calculateOutputValue();
	neuronAND.inputs[0].value = 0;
	neuronAND.inputs[1].value = 1;
	cout << "\n" << neuronAND.calculateOutputValue();
	neuronAND.inputs[0].value = 1;
	neuronAND.inputs[1].value = 1;
	cout << "\n" << neuronAND.calculateOutputValue();*/
	system("pause");
	return 0;
}