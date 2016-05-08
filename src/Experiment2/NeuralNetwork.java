import java.util.ArrayList;
import java.util.Random;
import java.io.PrintWriter;
import java.io.FileOutputStream;
import java.io.FileNotFoundException;

/**
 * Created by armandmaree on 2016/05/03.
 */
public class NeuralNetwork {
	private ArrayList<ArrayList<Double>> network;
	private ArrayList<ArrayList<ArrayList<Double>>> weights;
	private ArrayList<ArrayList<ArrayList<Double>>> deltaWeights;
	private char[] targets = {'\0'};
	private boolean hardMatch = true; // must match one of the given targets
	private int numCorrect = 0;
	private double trainingAccuracy = 0.99;
	private double  learningRate = 0.2;
	private double momentum = 0.2;
	private double meanSquareError = 0.0;
	private ArrayList<Double> mse;

	public NeuralNetwork (int[] topology) {
		Random generator = new Random();
		network = new ArrayList<>();
		weights = new ArrayList<>();
		deltaWeights = new ArrayList<>();
		mse = new ArrayList<>(3);

		for (int i = 0; i < 3; i++) {
			mse.add(0.0);
		}

		for (int i = 0; i < topology.length; i++) { // iterate through all the layers
			ArrayList<Double> layer;
			ArrayList<ArrayList<Double>> layerWeights; // list of each neuron's weights in the current layer.
			ArrayList<ArrayList<Double>> deltaLayerWeights; // list of each neuron's previous weights in the current layer.

			layer = new ArrayList<>(); // allocate extra space for extra neuron
			layerWeights = new ArrayList<>();
			deltaLayerWeights = new ArrayList<>();

			for (int j = 0; j < topology[i]; j++) { // iterate all neurons in curr layer
				layer.add(0.0);
			}

			if (i != topology.length - 1) { // add extra neuron to all levels except output
				layer.add(-1.0); // add extra neuron to layer

				for (int j = 0; j < topology[i] + 1; j++) { // iterate all neurons in current layer
					ArrayList<Double> neuronWeights = new ArrayList<>(); // the current neuron's weights
					ArrayList<Double> deltaNeuronWeights = new ArrayList<>(); // the current neuron's previous weights

					for (int k = 0; k < topology[i + 1]; k++) { // iterate all weights
						double bounds = 1 / (Math.sqrt(topology[i] + 1));
						neuronWeights.add(generator.nextDouble() * (2 * bounds) - bounds);
						deltaNeuronWeights.add(0.0);
					}

					layerWeights.add(neuronWeights); // allocate new list for each neuron in previous layer to current layer
					deltaLayerWeights.add(deltaNeuronWeights); // allocate new list for each neuron in previous layer to current layer
				}

				weights.add(layerWeights);
				deltaWeights.add(deltaLayerWeights);
			}

			network.add(layer); // add layer to network
		}
	}

	public void setTargets (char[] targets) {
		this.targets = targets;

		if (targets.length != network.get(network.size() - 1).size())
			hardMatch = false;
	}

	public void setNumCorrect (int numCorrect) {
		this.numCorrect = numCorrect;
	}

	public int getNumCorrect () {
		return numCorrect;
	}

	public void init() {
		numCorrect = 0;
		meanSquareError = 0.0;
	}

	public double[] getExtremeWeights() {
		double max = Double.MIN_VALUE;
		double min = Double.MAX_VALUE;

		for (int i = 0; i < weights.size(); i++) {
			for (int j = 0; j < weights.get(i).size(); j++) {
				for (int k = 0; k < weights.get(i).get(j).size(); k++) {
					if (weights.get(i).get(j).get(k) > max)
						max = weights.get(i).get(j).get(k);

					if(weights.get(i).get(j).get(k) < min)
						min = weights.get(i).get(j).get(k);
				}
			}
		}

		double[] tmp = {min, max};

		return tmp;
	}

	public void feedForward (ArrayList<Double> distortedImage) {
		char letter = (char)Math.floor(distortedImage.get(0));

		for (int i = 0; i < network.get(0).size() - 1; i++) { // set all the input values to the corresponding neurons except the last -1 neuron
			network.get(0).set(i, distortedImage.get(i + 1));
		}

		for (int i = 1; i < network.size(); i++) { // iterate through all the layers except the input layer
			for (int j = 0; j < network.get(i).size() - 1; j++) { // iterate through all the neurons in current layer except the -1 neuron at the bottom
				network.get(i).set(j, activation(i, j)); // set the value of neuron j at level i
			}
		}

		int outputLayerIndex = network.size() - 1;
		int lastOutputNeuronIndex = network.get(outputLayerIndex).size() - 1;
		network.get(outputLayerIndex).set(lastOutputNeuronIndex, activation(outputLayerIndex, lastOutputNeuronIndex)); // set the value of neuron j at level i

		int matchingIndex = getMatchingIndex(letter);
		boolean matches = true;

		for (int i = 0; i < network.get(outputLayerIndex).size(); i++) { // iterate all output neurons
			if (i == matchingIndex) {
				meanSquareError += Math.pow(1.0 - network.get(outputLayerIndex).get(i), 2);

				if (network.get(outputLayerIndex).get(i) < 0.7) {
					matches = false;
					break;
				}
			}
			else {
				meanSquareError += Math.pow(0.0 - network.get(outputLayerIndex).get(i), 2);

				if (network.get(outputLayerIndex).get(i) > 0.3) {
					matches = false;
					break;
				}
			}
		}

		if (matches) {
			numCorrect++; // letter was correctly and uniformly identified
		}
	}

	private double activation (int layer, int neuron) {
		double netVal = 0.0;

		for (int i = 0; i < network.get(layer - 1).size(); i++) { // iterate all the neurons i of neuron at layer = (layer - 1)
			netVal += network.get(layer - 1).get(i) * weights.get(layer - 1).get(i).get(neuron);
		}

		return 1 / (1 + Math.pow(Math.E, -1.0 * netVal));
	}

	private int getMatchingIndex(char letter) {
		int matchingIndex;

		if (!hardMatch) {
			matchingIndex = network.get(network.size() - 1).size() - 1; // default: last output neuron

			for (int i = 0; i < targets.length; i++) { // check if the letter of this data set is in our target list
				if (letter == targets[i]) { // if it is, the set that index as the one that must be 1
					matchingIndex = i;
					break;
				}
			}
		}
		else {
			matchingIndex = (int)letter - 65;
		}

		return matchingIndex;
	}

	public void adjustments() {
//		mse.remove(2);
//		mse.add(0, meanSquareError);
//
//		if (mse.get(0) < mse.get(1) && mse.get(1) < mse.get(2)) // if mean square error has decreased over the past 3 epoch
//			learningRate = Math.max(learningRate - 0.001, 0.01); // decrease learning rate up until 0.01
//		else if (mse.get(0) > mse.get(1) && mse.get(1) > mse.get(2)) // if mean square error has increased over the past 3 epoch
//			learningRate = Math.min(learningRate + 0.001, 1.0); // increase learning rate up until 1.0
//
//		momentum = Math.max(momentum - 0.01, 0.0); // decrease momentum up until 0
	}

	public double getMeanSquareError() {
		return meanSquareError;
	}

	public double getTrainingAccuracy() {
		return trainingAccuracy;
	}

	public void backPropagation (ArrayList<Double> distortedImage) {
		char letter = (char)Math.floor(distortedImage.get(0));
		int matchingIndex = getMatchingIndex(letter);
		int outputLayerIndex = network.size() - 1;
		ArrayList<Double> errorSignals = new ArrayList<>();

		for (int i = 0; i < network.get(outputLayerIndex).size(); i++) { // iterate all output neurons
			double target = 0.0; // default target for a neuron is 0.0
			double neuronValue = network.get(outputLayerIndex).get(i); // value of current neuron

			if (i == matchingIndex)
				target = 1.0; // if this neuron is the one that has to be true then set the target to 1.0

			double errorSignal = -1.0 * (target - neuronValue) * (1 - neuronValue) * neuronValue; // calculate the error signal of the current output
			errorSignals.add(errorSignal);

			for (int j = 0; j < network.get(outputLayerIndex - 1).size(); j++) { // iterate second last layer
				setNewWeights(outputLayerIndex - 1, j, i, errorSignal);
			}
		}

		for (int i = network.size() - 2; i > 0; i--) { // iterate all hidden layers
			ArrayList<Double> tmpErrorSignals = new ArrayList<>();

			for (int j = 0; j < network.get(i).size() - 1; j++) { // iterate all neurons at level at i
				double errorSignal = 0.0;
				int nextLayerLastNeuron = network.get(i + 1).size() - 1;

				for (int k = 0; k < nextLayerLastNeuron; k++) { // iterate neurons in next level to calculate error signal
					errorSignal += errorSignals.get(k) * weights.get(i).get(j).get(k) * (1 - network.get(i).get(j)) * network.get(i).get(j);
				}

				if (i == network.size() - 2)
					errorSignal += errorSignals.get(nextLayerLastNeuron) * weights.get(i).get(j).get(nextLayerLastNeuron) * (1 - network.get(i).get(j)) * network.get(i).get(j);

				tmpErrorSignals.add(errorSignal);

				for (int k = 0; k < network.get(i - 1).size(); k++) { // iterate previous layer
					setNewWeights(i - 1, k, j, errorSignal);
				}
			}

			errorSignals = tmpErrorSignals;
		}
	}

	private void setNewWeights(int layer, int neuron, int weight, double errorSignal) {
		double oldWeightKJ = weights.get(layer).get(neuron).get(weight);
		double valueOfK = network.get(layer).get(neuron);
		double oldDeltaKJ = deltaWeights.get(layer).get(neuron).get(weight);
		double newDeltaKJ = -1.0 * learningRate * errorSignal * valueOfK;
		double newWeightKJ = oldWeightKJ + newDeltaKJ + momentum * oldDeltaKJ;

		deltaWeights.get(layer).get(neuron).set(weight, newDeltaKJ);
		weights.get(layer).get(neuron).set(weight, newWeightKJ);
	}

	public void log(long epoch, double trainCorrect, double generalCorrect) {
		PrintWriter out = null;

		try {
			out = new PrintWriter(new FileOutputStream("NNStats.txt", true));
		}
		catch(FileNotFoundException e) {
			System.out.println("'NNStats.txt' could not be found.");
			e.printStackTrace();
			System.exit(1);
		}

		out.println("Epoch: " + epoch);
		out.println("\tMSE: " + meanSquareError);
		out.println("\tTraining Correct: " + (Math.round (trainCorrect * 100000.0) / 1000.0) + "%");
		out.println("\tGeneral Correct: " + (Math.round (generalCorrect * 100000.0) / 1000.0) + "%");

		out.close();
	}

	public void logCSV(long epoch, double trainCorrect, double generalCorrect) {
		PrintWriter out = null;

		try {
			out = new PrintWriter(new FileOutputStream("results.csv", true));
		}
		catch(FileNotFoundException e) {
			System.out.println("'results.csv' could not be found.");
			e.printStackTrace();
			System.exit(1);
		}

		out.printf("" + epoch);
		out.printf("," + (100 - Math.round (trainCorrect * 100000.0) / 1000.0));
		out.printf("," + (100 - Math.round (generalCorrect * 100000.0) / 1000.0));
		out.printf("," + (network.size() - 2));
		out.printf("," + learningRate);
		out.println("," + momentum);

		out.close();
	}
}
