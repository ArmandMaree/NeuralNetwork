#include <iostream>
#include <ctime>
#include <cmath>
#include <cstdlib>

#include "Neuron.h"

using namespace std;

double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

Neuron::Neuron(vector<Neuron*>* inputNeurons, vector<Neuron*>* outputNeurons){
	this->inputNeurons = inputNeurons;
	this->outputNeurons = outputNeurons;
	recursiveDelete = false;
	momentum = 0;
	lastEpoch = 0;
	weights = new vector<double>();

	if (inputNeurons != NULL) {
		srand(time(NULL));
		double maxmin = 1.0 / sqrt(inputNeurons->size());

		for (size_t i = 0; i < inputNeurons->size() - 1; i++)
			weights->push_back(fRand(-maxmin, maxmin));

		weights->push_back(1.0);
	}
	else
		weights = NULL;
}

Neuron::~Neuron(){
	if (outputNeurons != NULL && recursiveDelete) {
		for (size_t i = 0; i < outputNeurons->size(); i++) {
			if (outputNeurons->at(i) != NULL) {
				if (i == 0)
					outputNeurons->at(i)->recursiveDelete = true;

				delete outputNeurons->at(i);
			}
		}

		delete outputNeurons;
	}

	if (weights != NULL)
		delete weights;

	outputNeurons = NULL;
}

void Neuron::setName(string name) {
	this->name = name;
}

string Neuron::getName() {
	return name;
}
