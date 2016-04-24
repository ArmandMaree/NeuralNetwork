#ifndef NEURON_H
#define NEURON_H

#include <iostream>
#include <vector>
#include <string>

using namespace std;

class Neuron {
private:
	vector<Neuron*>* inputNeurons;
	vector<Neuron*>* outputNeurons;
	string name;
	vector<double>* weights;
	int momentum;
	int lastEpoch;
	double output;
protected:

public:
	bool recursiveDelete;
	Neuron(vector<Neuron*>* inputNeurons, vector<Neuron*>* outputNeurons);
	~Neuron();
	void setName(string name);
	string getName();
};

#endif
