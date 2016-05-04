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
	private double momentum = 0.9;
	private double meanSquareError = 0.0;
	private ArrayList<Double> mse = new ArrayList<>(3);

	public NeuralNetwork (int[] topology) {
		Random generator = new Random();
		network = new ArrayList<>();
		weights = new ArrayList<>();
		deltaWeights = new ArrayList<>();

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
//			System.out.println("MATCHED LETTER '" + letter + "' at index " + matchingIndex);
			numCorrect++; // letter was correctly and uniformly identified
		}

//		if (matches && (letter == 'A' || letter == 'E' || letter == 'O' || letter == 'U' || letter == 'I'))
//			System.out.println("MATCHED LETTER '" + letter + "' at index " + matchingIndex);
//		else if (matches)
//			System.out.println("NOT MATCHED LETTER '" + letter + "' at index " + matchingIndex);
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
		int outputLayerIndex = 2;
		ArrayList<Double> errorSignals = new ArrayList<>();

		for (int i = 0; i < network.get(outputLayerIndex).size(); i++) { // iterate all output neurons
			double target = 0.0; // default target for a neuron is 0.0
			double neuronValue = network.get(outputLayerIndex).get(i); // value of current neuron

			if (i == matchingIndex)
				target = 1.0; // if this neuron is the one that has to be true then set the target to 1.0

			double errorSignal = -1.0 * (target - neuronValue) * (1 - neuronValue) * neuronValue; // calculate the error signal of the current output
			errorSignals.add(errorSignal);

			for (int j = 0; j < network.get(outputLayerIndex - 1).size(); j++) { // iterate second last layer
				double oldWeightJI = weights.get(outputLayerIndex - 1).get(j).get(i);
				double valueOfJ = network.get(outputLayerIndex - 1).get(j);
				double oldDeltaJI = deltaWeights.get(outputLayerIndex - 1).get(j).get(i);
				double newDeltaJI = -1.0 * learningRate * errorSignal * valueOfJ;
				double newWeightJI = oldWeightJI + newDeltaJI + momentum * oldDeltaJI;

				deltaWeights.get(outputLayerIndex - 1).get(j).set(i, newDeltaJI);
				weights.get(outputLayerIndex - 1).get(j).set(i, newWeightJI);
			}
		}



		for (int i = network.size() - 2; i > 0; i--) { // iterate all hidden layers
			ArrayList<Double> tmpErrorSignals = null;

			for (int j = 0; j < network.get(i).size() - 1; j++) { // iterate all neurons at level at i
				tmpErrorSignals = new ArrayList<>();
				double errorSignal = 0.0;

				for (int k = 0; k < network.get(i + 1).size(); k++) { // iterate neurons in next level to calculate error signal
					errorSignal += errorSignals.get(k) * weights.get(i).get(j).get(k) * (1 - network.get(i).get(j)) * network.get(i).get(j);
				}

				tmpErrorSignals.add(errorSignal);

				for (int k = 0; k < network.get(i - 1).size(); k++) { // iterate previous layer
					//System.out.println(weights.get(i - 1).get(k).size());
					double oldWeightKJ = weights.get(i - 1).get(k).get(j);
					double valueOfK = network.get(i - 1).get(k);
					double oldDeltaKJ = deltaWeights.get(i - 1).get(k).get(j);
					double newDeltaKJ = -1.0 * learningRate * errorSignal * valueOfK;
					double newWeightKJ = oldWeightKJ + newDeltaKJ + momentum * oldDeltaKJ;

					deltaWeights.get(i - 1).get(k).set(j, newDeltaKJ);
					weights.get(i - 1).get(k).set(j, newWeightKJ);
				}
			}

			errorSignals = tmpErrorSignals;
		}
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
}
