#include "MulipleOutputTrainer.h"
#include <iostream>
#include <algorithm>
#include <chrono>
MultipleOutputTrainer::MultipleOutputTrainer(vector<pair<vector<double>, vector<vector<double>>>> samples, Network & network)
{
	double ni = 0.05;
	vector<double> deltas;
	double delta = 1;
	int counter = 0;
	double error = 0.05;
	while (counter < 40)
	{
		typedef std::chrono::high_resolution_clock Time;
		typedef std::chrono::milliseconds ms;
		typedef std::chrono::duration<float> fsec;
		auto t0 = Time::now();
		
		for (auto eachSample : samples)
		{
			for (vector<double> sample : eachSample.second)
			{
				network.initializeInputs(sample, 0);
				network.feedForward();
				network.backPropagation(eachSample.first);
				vector<double> eachTarget = eachSample.first;
				if (counter > 55)
				{
					for (int j = 0; j < eachTarget.size(); ++j)
					{
						delta += pow(eachTarget[j] - network.layers.back()[j].getOutputValue(), 2);
					}
					cout << "\n delta: " << 0.5*delta << endl;
					delta = 0;
				}
				
				
			}
		}
		auto t1 = Time::now();
		fsec fs = t1 - t0;
		
		cout << "Epoc: " << counter << " \t time: " << fs.count() << "s" << endl;
		counter++;
	}
}

void MultipleOutputTrainer::displayImg(vector<double> image)
{
	int m = 0;
	for (int j = 0; j < 30; ++j)
	{
		for (int k = 0; k < 20; ++k)
		{
			cout << image[m];
			m++;
		}
		cout << endl;
	}
}



bool MultipleOutputTrainer::ifErrorsSmallEnought(vector<vector<double>> inputs, vector<vector<double>> targetVal, Network & network, double error)
{
	double delta = 0;
	for (vector<double> eachTarget : targetVal) 
	{
		for (int j = 0; j < eachTarget.size(); ++j)
		{
			delta += pow(eachTarget[j] - network.layers.back()[j].getOutputValue(), 2);
			if (0.5*delta > error)
			{
				cout << "\n delta: " << 0.5*delta << endl;
				return false;
			}
		}
	}
	return true;
}