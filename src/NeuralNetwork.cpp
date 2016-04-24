#include <string>
#include <sstream>

#include "NeuralNetwork.h"

using namespace std;

NeuralNetwork::NeuralNetwork(int* topology, int numLevels){
	vector<Neuron*>* prev = NULL;
	vector<Neuron*>* next = new vector<Neuron*>();
	epochMax = 10000;
	epochCounter = 0;
	learningRate = 0.5;
	trainingError = 0.0;

   	for (size_t i = 0; i < numLevels; i++) {
		vector<Neuron*>* level = next;

		if (i != numLevels - 1)
			next = new vector<Neuron*>();
		else
			next = NULL;

	   	for (size_t j = 0; j < *(topology + i); j++) {
			Neuron* tmp = new Neuron(prev, next);
			stringstream ssName;
			ssName << "[" << i << "," << j << "]";
			tmp->setName(ssName.str());
		   	level->push_back(tmp);
	   	}

		if (i != numLevels - 1) {
			Neuron* tmp = new Neuron(prev, next);
			stringstream ssName;
			ssName << "[" << i << "," << *(topology + i) << "]";
			tmp->setName(ssName.str());
			level->push_back(tmp);
		}

		prev = level;

		if (i == 0)
			inputNeurons = level;

		if (i == numLevels - 1)
			outputNeurons = level;
   }
}

NeuralNetwork::~NeuralNetwork() {
	if (inputNeurons != NULL) {
		for (size_t i = 0; i < inputNeurons->size(); i++) {
			if (inputNeurons->at(i) != NULL) {
				if (i == 0)
					inputNeurons->at(i)->recursiveDelete = true;

				delete inputNeurons->at(i);
			}
		}

		delete inputNeurons;
	}
}
