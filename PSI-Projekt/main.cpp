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
	vector<double> vec{ 2, 4, 1 };
	Network network(vec);
	//network.initializeInputs(input, 0);
	//network.feedForward();

	//xor
	vector<vector<double>> neurons = { {0,0}, {0,1}, {1,0}, {1,1} };
	vector<double> targetVal = { 0,1,1,0};
	
	BaseTrainer trainer(neurons, targetVal, network);
	vector<double> inp = { 0,1 };
	network.initializeInputs(inp, 0);
	network.feedForward();
	cout << "\nTest : " << inp[0] << " " << inp[1] << "\t" << network.layers.back().back().getOutputValue();

	inp = { 1,0 };
	network.initializeInputs(inp, 0);
	network.feedForward();
	cout << "\nTest : " << inp[0] << " " << inp[1] << "\t" << network.layers.back().back().getOutputValue();


	inp = { 0,0 };
	network.initializeInputs(inp, 0);
	network.feedForward();
	cout << "\nTest : " << inp[0] << " " << inp[1] << "\t" << network.layers.back().back().getOutputValue();


	inp = { 1,1 };
	network.initializeInputs(inp, 0);
	network.feedForward();
	cout << "\nTest : " << inp[0] << " " << inp[1] << "\t" << network.layers.back().back().getOutputValue();

	system("pause");
	return 0;
}