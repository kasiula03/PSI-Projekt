#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include "Neuron.h"
#include "BaseTrainer.h"
using namespace std;

int main()
{
	srand(time(NULL));

	vector<Neuron> neurons;
	vector<double> targetVal;
	//and
	for (int i = 2000; i >= 0; --i)
	{
		int n1 = (int)(2.0 * rand() / double(RAND_MAX));
		int n2 = (int)(2.0 * rand() / double(RAND_MAX));

		int t = n1 & n2;
		Neuron neuron(2);
		neuron.inputs[0].value = n1;
		neuron.inputs[1].value = n2;
		neurons.push_back(neuron);
		targetVal.push_back(t);
		//cout << n1 << " " << n2 << " " << t << endl;
	}
	
	//BaseTrainer trainer(neurons, targetVal);
	//trainer.weigthTest(neurons, targetVal);

	//or
	vector<Neuron> neurons2;
	vector<double> targetVal2;
	for (int i = 2000; i >= 0; --i)
	{
		int n1 = (int)(2.0 * rand() / double(RAND_MAX));
		int n2 = (int)(2.0 * rand() / double(RAND_MAX));

		int t = n1 | n2;
		Neuron neuron(2);
		neuron.inputs[0].value = n1;
		neuron.inputs[1].value = n2;
		neurons2.push_back(neuron);
		targetVal2.push_back(t);
		//cout << n1 << " " << n2 << " " << t << endl;
	}

	//BaseTrainer trainer2(neurons2, targetVal2);
	//trainer2.weigthTest(neurons2, targetVal2);
	//not
	vector<Neuron> neurons3;
	vector<double> targetVal3;
	for (int i = 2000; i >= 0; --i)
	{
		int n1 = (int)(2.0 * rand() / double(RAND_MAX));

		int t = !n1;
		Neuron neuron(1);
		neuron.inputs[0].value = n1;
		neurons3.push_back(neuron);
		targetVal3.push_back(t);
		//cout << n1 << " " << n2 << " " << t << endl;
	}

	BaseTrainer trainer3(neurons3, targetVal3);

	system("pause");
	return 0;
}