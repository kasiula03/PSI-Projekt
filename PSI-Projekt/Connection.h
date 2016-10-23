#pragma once

class Neuron;

class Connection
{
public:
	double weight;
	double deltaWeight;
	double value;
	Neuron * input;
};