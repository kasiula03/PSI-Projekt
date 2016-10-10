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

	vector<vector<double>> neurons;
	vector<double> targetVal;
	//and
	Neuron neuron1(2);
	for (int i = 2000; i >= 0; --i)
	{
		int n1 = (int)(2.0 * rand() / double(RAND_MAX));
		int n2 = (int)(2.0 * rand() / double(RAND_MAX));

		int t = n1 & n2;
		vector<double> neuron;
		neuron.push_back(n1);
		neuron.push_back(n2);
		neurons.push_back(neuron);
		targetVal.push_back(t);
		//cout << n1 << " " << n2 << " " << t << endl;
	}
	
	BaseTrainer trainer(neurons, targetVal, neuron1);
	Neuron test1(2);
	test1.inputs[0].weight = neuron1.inputs[0].weight;
	test1.inputs[1].weight = neuron1.inputs[1].weight;
	test1.inputs[0].value = 0;
	test1.inputs[1].value = 0;
	Neuron test2(2);
	test2.inputs[0].weight = neuron1.inputs[0].weight;
	test2.inputs[1].weight = neuron1.inputs[1].weight;
	test2.inputs[0].value = 1;
	test2.inputs[1].value = 0;
	Neuron test3(2);
	test3.inputs[0].weight = neuron1.inputs[0].weight;
	test3.inputs[1].weight = neuron1.inputs[1].weight;
	test3.inputs[0].value = 0;
	test3.inputs[1].value = 1;
	Neuron test4(2);
	test4.inputs[0].weight = neuron1.inputs[0].weight;
	test4.inputs[1].weight = neuron1.inputs[1].weight;
	test4.inputs[0].value = 1;
	test4.inputs[1].value = 1;
	cout << endl << endl << test1.calculateOutputValue() << " " << test2.calculateOutputValue() << " " << test3.calculateOutputValue() << " " << test4.calculateOutputValue() << endl;
	//trainer.weigthTest(neurons, targetVal);

	//or
	Neuron neuron2(2);
	vector<vector<double>> neurons2;
	vector<double> targetVal2;
	for (int i = 2000; i >= 0; --i)
	{
		int n1 = (int)(2.0 * rand() / double(RAND_MAX));
		int n2 = (int)(2.0 * rand() / double(RAND_MAX));

		int t = n1 | n2;
		vector<double> neuron;
		neuron.push_back(n1);
		neuron.push_back(n2);
		neurons2.push_back(neuron);
		targetVal2.push_back(t);
		//cout << n1 << " " << n2 << " " << t << endl;
	}

	BaseTrainer trainer2(neurons2, targetVal2, neuron2);
	test1.inputs[0].weight = neuron2.inputs[0].weight;
	test1.inputs[1].weight = neuron2.inputs[1].weight;
	test2.inputs[0].weight = neuron2.inputs[0].weight;
	test2.inputs[1].weight = neuron2.inputs[1].weight;
	test3.inputs[0].weight = neuron2.inputs[0].weight;
	test3.inputs[1].weight = neuron2.inputs[1].weight;
	test4.inputs[0].weight = neuron2.inputs[0].weight;
	test4.inputs[1].weight = neuron2.inputs[1].weight;
	cout << endl << endl << test1.calculateOutputValue() << " " << test2.calculateOutputValue() << " " << test3.calculateOutputValue() << " " << test4.calculateOutputValue() << endl;

	//trainer2.weigthTest(neurons2, targetVal2);
	//not
	vector<vector<double>> neurons3;
	vector<double> targetVal3;
	Neuron neuron3(1);
	for (int i = 2000; i >= 0; --i)
	{
		int n1 = (int)(2.0 * rand() / double(RAND_MAX));

		int t = !n1;
		vector<double> neuron;
		neuron.push_back(n1);
		neurons3.push_back(neuron);
		targetVal3.push_back(t);
		
		//cout << n1 << " " << n2 << " " << t << endl;
	}

	BaseTrainer trainer3(neurons3, targetVal3, neuron3);
	Neuron t1(1), t2(1), t3(1), t4(1);
	t1.inputs[0].weight = neuron3.inputs[0].weight;
	t2.inputs[0].weight = neuron3.inputs[0].weight;
	t1.inputs[0].value = 0;
	t2.inputs[0].value = 1;

	cout << endl << endl << t1.calculateOutputValue() << " " << t2.calculateOutputValue() << endl;

	system("pause");
	return 0;
}