#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <iostream>
#include <vector>

using namespace std;

#include "Neuron.h"

class NeuralNetwork {
private:
	int* topology;
	vector<Neuron*>* inputNeurons;
	vector<Neuron*>* outputNeurons;
	unsigned int epochCounter;
	unsigned int epochMax;
	double learningRate;
	double trainingError;
	double trainingAccuracy;
protected:

public:
	NeuralNetwork(int* topology, int numLevel);
	~NeuralNetwork();
	void feedForward(vector<double>* dataSet);
};

#endif
