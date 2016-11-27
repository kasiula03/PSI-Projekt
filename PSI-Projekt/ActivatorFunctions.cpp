#include "ActivatorFunctions.h"
#include <cmath>

double ActivatorFunctions::sigmoid(double x)
{
	double mian = 1 + exp(-x);
	return (1 / mian);
}

double ActivatorFunctions::derivativeSigmoid(double x)
{
	return ActivatorFunctions::sigmoid(x)*(1.0 - ActivatorFunctions::sigmoid(x));
}