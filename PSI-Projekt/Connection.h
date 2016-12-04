#pragma once

class Neuron;

class Connection
{
public:
	double weight;
	double value;
	Neuron * input;
};