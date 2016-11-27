#pragma once
#include <vector>
#include "Network.h"

using namespace std;

class MultipleOutputTrainer
{
public:
	MultipleOutputTrainer(vector<vector<double>>, vector<vector<double>>, Network &);
	bool ifErrorsSmallEnought(vector<vector<double>> inputs, vector<vector<double>> targetVal, Network & network, double error);
};