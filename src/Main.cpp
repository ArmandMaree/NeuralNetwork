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

void shuffleImages(vector<vector<int>*>* distortedImages);
void cleanVector(vector<vector<int>*>* distortedImages);

int main (int argc, char* argv[]) {
	string line;
  	ifstream dataFile("./images/letter-recognition.data");
	int numInputs;
	bool read = false;
	vector<vector<int>*> distortedImages;

	if (dataFile.is_open()) {
		cout << "Reading data..." << endl;

		while (getline(dataFile,line)) {
			vector<int>* vect = new vector<int>();
			stringstream ss(line);
			int i;
			char letter;
			ss >> letter;
			vect->push_back((int)letter); // read letter
			ss.ignore(); // ignore first comma

			while (ss >> i) {
				vect->push_back(i);

				if (ss.peek() == ',')
					ss.ignore();
			}

			numInputs = vect->size();

			if (!read) {
				int topology[] = {numInputs, numInputs, 26};
				NeuralNetwork nn(topology, 3);
			}

			distortedImages.push_back(vect);
		}

		dataFile.close();

		cout << "Shuffling data..." << endl;
		shuffleImages(&distortedImages);

		cout << "Cleaning memory..." << endl;
		cleanVector(&distortedImages);
	}
	else cout << "Unable to open file";

	return 0;
}

void shuffleImages(vector<vector<int>*>* distortedImages) {
	for (int i = 0; i < distortedImages->size(); i++) {
		int pos = (int)rand() % distortedImages->size();
		vector<int>* tmp = distortedImages->at(pos);
		distortedImages->erase(distortedImages->begin() + pos);
		distortedImages->push_back(tmp);
	}
}

void cleanVector(vector<vector<int>*>* distortedImages) {
	int size = distortedImages->size();
	for (int i = 0; i < size; i++) {
		delete distortedImages->at(0);
		distortedImages->erase(distortedImages->begin());
	}
}
