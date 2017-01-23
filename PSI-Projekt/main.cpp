#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <map>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "ImageConverter.h"
#include "BaseTrainer.h"
#include "MulipleOutputTrainer.h"
#include "ActivatorFunctions.h"
#include "Network.h"

using namespace cv;
using namespace std;

void xor();
vector<double> getImgAsFloats(Mat image);
vector<pair<vector<double>, vector<vector<double>>>> prepareTrainingSamples();
map<char, vector<vector<double>>> prepareMapSamples();
void displayImg(vector<double> image);
void testLetter(Mat im, Network & network);
char testLetter(vector<double> vec, Network & network);
vector<pair<Rect, vector<double>>> sortLetter(vector<pair<Rect, vector<double>>>);
int main()
{
	srand(time(NULL));

	Mat img1 = cv::imread("test_letter.jpg");
	Mat img2 = cv::imread("test_letter0.jpg");
	Mat img3 = cv::imread("test_letterS.png");
	Mat test = cv::imread("testIMG.png");

	Neuron::activator = ActivatorFunctions::sigmoid;
	Neuron::derivativeActivator = ActivatorFunctions::derivativeSigmoid;
	ImageConverter converter;
	Mat traingImg = imread("training_chars0.png");
	vector<pair<vector<double>, vector<vector<double>>>> map = prepareTrainingSamples();
	vector<vector<double>> targets;
	vector<vector<double>> inputs;
	/*for (pair<vector<double>, vector<vector<double>>> pair : map)
	{
		for (vector<double> sample : pair.second)
		{
			displayImg(sample);
		}
	}*/
	for (pair<vector<double>, vector<vector<double>>> eachSample : map)
	{
		targets.push_back(eachSample.first);
		inputs.insert(inputs.end(), eachSample.second.begin(), eachSample.second.end());
	}
	
	//for (int i = 0; i < map.size(); ++i)
		//displayImg(map[i]);

	double inputsSize = inputs[0].size();
	double outputsSize = targets[0].size();
	vector<double> vec{inputsSize, sqrt(inputsSize*outputsSize), outputsSize };
	Network network(vec);
	network.loadNetwork("weights_1483837191.txt");
	auto let = ImageConverter::prepareImg(test);
	vector<Mat> letters;
	vector<vector<double>> letterConv;
	vector<pair<Rect, vector<double>>> sortedLetter = sortLetter(let);
	string text;
	for (auto letter : sortedLetter)
	{
		text += testLetter(letter.second, network);
		cout << endl;
	}
	cout << text << endl;
	testLetter(img1, network);
	testLetter(img2, network);
	testLetter(img3, network);


	MultipleOutputTrainer multipleTrainer(map, network);
	//network.saveWeights();

	

	system("pause");
	return 0;
}

vector<pair<Rect, vector<double>>> sortLetter(vector<pair<Rect, vector<double>>> letters)
{
	int n = letters.size();
	do
	{
		for (int i = 0; i < n - 1; i++)
		{
			if (letters[i].first.x > letters[i + 1].first.x && letters[i].first.y <= letters[i + 1].first.y)
				swap(letters[i], letters[i + 1]);
		}
		n--;
	} while (n > 1);
	return letters;
	
}

vector<double> getImgAsFloats(Mat image)
{
	vector<vector<double>> sample = ImageConverter::prepareSamples(image);
	
	return sample[0];
}

void xor()
{
	Neuron::activator = ActivatorFunctions::sigmoid;
	Neuron::derivativeActivator = ActivatorFunctions::derivativeSigmoid;
	vector<double> vec{ 2, 4, 1 };
	Network network(vec);

	//xor
	vector<vector<double>> neurons = { { 0,0 },{ 0,1 },{ 1,0 },{ 1,1 } };
	vector<double> targetVal = { 0,1,1,0 };

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
}

char testLetter(vector<double> vec, Network & network)
{
	double best = 0;
	char chosen;
	network.initializeInputs(vec, 0);
	network.feedForward();
	vector<Neuron> outputLayer = network.layers[1];
	cout << "\n\n\nResults\n\n";
	displayImg(vec);
	for (int i = 0; i < outputLayer.size(); ++i)
		cout << "neuron: " << (char)(i + 65) << "\t" << outputLayer[i].getOutputValue() << endl;

	for (int i = 0; i < outputLayer.size(); ++i)
	{
		if (best < outputLayer[i].getOutputValue())
		{
			best = outputLayer[i].getOutputValue();
			chosen = (char)(i + 65);
		}
	}
	return chosen;
}

void testLetter(Mat im, Network & network)
{
	vector<double> img = getImgAsFloats(im);
	network.initializeInputs(img, 0);
	network.feedForward();
	vector<Neuron> outputLayer = network.layers[1];
	cout << "\n\n\nResults\n\n";
	displayImg(img);
	for (int i = 0; i < outputLayer.size(); ++i)
		cout << "neuron: " << (char)(i + 65) << "\t" << outputLayer[i].getOutputValue() << endl;

}

void displayImg(vector<double> image)
{
	int m = 0;
	for (int j = 0; j < 21; ++j)
	{
		for (int k = 0; k < 14; ++k)
		{
			cout << image[m];
			m++;
		}
		cout << endl;
	}
}

vector<pair<vector<double>, vector<vector<double>>>> prepareTrainingSamples()
{
	map<char, vector<vector<double>>> samples = prepareMapSamples();
	vector<pair<vector<double>, vector<vector<double>>>> convertedSamples;
	int size = samples.size();
	for (pair<char, vector<vector<double>>> eachSample : samples)
	{
		vector<double> target;
		int key = (int)eachSample.first - 65;
		for (int i = 0; i < 26; i++)
		{
			if (i == key)
				target.push_back(1);
			else
				target.push_back(0);
		}
		convertedSamples.push_back(pair<vector<double>, vector<vector<double>>>(target, eachSample.second));
	}
	return convertedSamples;
}

map<char, vector<vector<double>>> prepareMapSamples()
{
	map<char, vector<vector<double>>> samples;
	for (int i = 65; i < 91; i++)
	{
		char sign = (char)i;
		string fileName;
		fileName.push_back(sign);
		fileName += ".png";
		Mat img1 = cv::imread("TrainingSample/" + fileName);
		
		vector<vector<double>> letterSample = ImageConverter::prepareSamples(img1);
		samples.insert(pair<char, vector<vector<double>>>(sign, letterSample));
	}
	return samples;
}