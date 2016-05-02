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
	double value;
	double threshhold;
protected:

public:
	bool recursiveDelete;
	Neuron(vector<Neuron*>* inputNeurons, vector<Neuron*>* outputNeurons);
	~Neuron();
	void setName(string name);
	string getName();
	void setValue(string name);
	string getValue();
	double feedForward(int epoch);
};

#endif
