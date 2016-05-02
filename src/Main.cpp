#include <iostream>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <ctime>
#include <cmath>

#include "NeuralNetwork.h"

using namespace std;

void normalizeData(vector<vector<double>*>* distortedImages, vector<double>* min, vector<double>* max);
void shuffleImages(vector<vector<double>*>* distortedImages);
void cleanDVector(vector<vector<double>*>* distortedImages);

int main (int argc, char* argv[]) {
	string line;
  	ifstream dataFile("./images/letter-recognition.data");
	int numInputs;
	bool read = false;
	vector<vector<double>*> distortedImages;
	vector<double> min;
	vector<double> max;
	NeuralNetwork* nn;

	if (dataFile.is_open()) {
		cout << "Reading data..." << endl;

		vector<double> min;
		vector<double> max;

		for (size_t i = 0; i < 16; i++) {
			min.push_back(0.0);
			max.push_back(0.0);
		}

		while (getline(dataFile,line)) {
			vector<double>* vect = new vector<double>();
			stringstream ss(line);
			int i;
			char letter;
			ss >> letter;
			vect->push_back((double)letter); // read letter
			ss.ignore(); // ignore first comma
			int counter = 0;

			while (ss >> i) {
				vect->push_back(i);


				if ((double)i < min.at(vect->size() - 2))
					min.at(vect->size() - 2) = (double)i;

				if ((double)i > max.at(vect->size() - 2))
					max.at(vect->size() - 2) = (double)i;

				if (ss.peek() == ',')
					ss.ignore();
			}

			numInputs = vect->size();

			if (!read) {
				int topology[] = {numInputs, numInputs, 26};
				nn = new NeuralNetwork(topology, 3);
				read = true;
			}

			distortedImages.push_back(vect);
		}

		dataFile.close();
		shuffleImages(&distortedImages);
		normalizeData(&distortedImages, &min, &max);

		for (size_t i = 0; i < distortedImages.size() * 0.6; i++) {
			nn->feedForward(distortedImages.at(i));
		}

		delete nn;
		cleanDVector(&distortedImages);
	}
	else cout << "Unable to open file";

	return 0;
}

void normalizeData(vector<vector<double>*>* distortedImages, vector<double>* min, vector<double>* max) {
	cout << "Normalizing data..." << endl;

	for (size_t i = 0; i < distortedImages->size(); i++) {
		for (size_t j = 1; j < distortedImages->at(i)->size() - 1; j++) {
			double tmp = distortedImages->at(i)->at(j);
			distortedImages->at(i)->at(j) = (tmp - min->at(j)) / (max->at(j) - min->at(j));
		}
	}
}

void shuffleImages(vector<vector<double>*>* distortedImages) {
	cout << "Shuffling data..." << endl;

	for (int i = 0; i < distortedImages->size(); i++) {
		int pos = (int)rand() % distortedImages->size();
		vector<double>* tmp = distortedImages->at(pos);
		distortedImages->erase(distortedImages->begin() + pos);
		distortedImages->push_back(tmp);
	}
}

void cleanDVector(vector<vector<double>*>* dVect) {
	cout << "Cleaning memory..." << endl;

	int size = dVect->size();
	for (int i = 0; i < size; i++) {
		delete dVect->at(0);
		dVect->erase(dVect->begin());
	}
}
