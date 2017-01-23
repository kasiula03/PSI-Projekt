#pragma once
#include <vector>
#include <map>
#include "Network.h"

using namespace std;

class MultipleOutputTrainer
{
public:
	MultipleOutputTrainer(vector<pair<vector<double>, vector<vector<double>>>>, Network &);
	void displayImg(vector<double> image);
	void saveWeights();
	bool ifErrorsSmallEnought(vector<vector<double>> inputs, vector<vector<double>> targetVal, Network & network, double error);
};