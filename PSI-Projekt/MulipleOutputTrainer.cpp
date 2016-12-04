#include "MulipleOutputTrainer.h"
#include <iostream>
#include <algorithm>
#include <chrono>
MultipleOutputTrainer::MultipleOutputTrainer(vector<vector<double>> inputs, vector<vector<double>> targetVal, Network & network)
{
	double ni = 0.05;
	int inputCounter = inputs.back().size();
	vector<double> deltas;
	double delta = 1;
	int counter = 0;
	int multiply = inputs.size() / targetVal.size();
	vector<double> currentInputs = inputs[0];
	network.initializeInputs(currentInputs, 0);
	network.feedForward();
	
	while (counter < 70)
	{
		typedef std::chrono::high_resolution_clock Time;
		typedef std::chrono::milliseconds ms;
		typedef std::chrono::duration<float> fsec;
		auto t0 = Time::now();
		
		for (int i = 0; i < inputs.size(); i++)
		{
			vector<double> currentInputs = inputs[i];
			network.initializeInputs(currentInputs, 0);
			network.feedForward();
			/*for (int j = 0; j < targetVal[i].size(); ++j)
			{
				delta = targetVal[i][j] - network.layers.back()[j].getOutputValue();
				cout << "Blad" << abs(delta) << "\t target: " << targetVal[i][j] << " get: " << network.layers.back()[j].getOutputValue() << endl;
			}
			cout << "\n Input" << i << endl;*/
			//deltas.push_back(delta);
			
			//showResult(network);
		
			network.backPropagation(targetVal[i]); 
		}
		auto t1 = Time::now();
		fsec fs = t1 - t0;
		
		cout << "Epoc: " << counter << " \t time: " << fs.count() << "s" << endl;
		counter++;
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