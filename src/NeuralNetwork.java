import java.util.ArrayList;
import java.util.Random;

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
	private double  learningRate = 0.1;
	private double momentum = 0.1;
	private int epochCounter;
	private int maxEpoch = 20000;

	public NeuralNetwork (int[] topology) {
		network = new ArrayList<>(topology.length);
		Random generator = new Random();
		weights = new ArrayList<>(topology.length);
		deltaWeights = new ArrayList<>(topology.length);

		for (int i = 0; i < topology.length; i++) { // iterate through all the layers
			ArrayList<Double> layer;
			ArrayList<ArrayList<Double>> layerNeurons;
			ArrayList<ArrayList<Double>> deltaLayerNeurons;

			if (i != topology.length - 1) { // if not the output layer
				layer = new ArrayList<>(topology[i] + 1); // allocate extra space for extra neuron
				layerNeurons = new ArrayList<>(topology[i] + 1); // list of each neuron's weights in the current layer.
				deltaLayerNeurons = new ArrayList<>(topology[i] + 1); // list of each neuron's previous weights in the current layer.
			}
			else {
				layer = new ArrayList<>(topology[i]);
				layerNeurons = new ArrayList<>(topology[i]); // list of each neuron's weights in the current layer.
				deltaLayerNeurons = new ArrayList<>(topology[i]); // list of each neuron's previous weights in the current layer.
			}

			for (int j = 0; j < topology[i]; j++) {
				layer.add(0.0);

				if (i != 0) { // if not the input layer
					ArrayList<Double> neuronWeights = new ArrayList<>(network.get(i - 1).size()); // the current neuron's weight excluding input layer
					ArrayList<Double> deltaNeuronWeights = new ArrayList<>(network.get(i - 1).size()); // the current neuron's previous weight excluding input layer

					for (int k = 0; k < network.get(i - 1).size(); k++) {
						neuronWeights.add(generator.nextDouble() * (2 * (topology[i - 1] + 1)) - (topology[i - 1] + 1));
						deltaNeuronWeights.add(0.0);
					}

					layerNeurons.add(neuronWeights); // allocate new list for each neuron in previous layer to current layer
					deltaLayerNeurons.add(deltaNeuronWeights); // allocate new list for each neuron in previous layer to current layer

					if (j == topology[i] - 1) {
						neuronWeights = new ArrayList<>(network.get(i - 1).size()); // the current neuron's weight excluding input layer
						deltaNeuronWeights = new ArrayList<>(network.get(i - 1).size()); // the current neuron's previous weight excluding input layer

						for (int k = 0; k < network.get(i - 1).size(); k++) {
							neuronWeights.add(generator.nextDouble() * (2 * (topology[i - 1] + 1)) - (topology[i - 1] + 1));
							deltaNeuronWeights.add(0.0);
						}

						layerNeurons.add(neuronWeights); // allocate new list for each neuron in previous layer to current layer
						deltaLayerNeurons.add(deltaNeuronWeights); // allocate new list for each neuron in previous layer to current layer
					}
				}
			}

			if (i != topology.length - 1) { // if not the output layer
				layer.add(-1.0); // add extra neuron to layer
			}

			if (i != 0) { // if not the input layer
				weights.add(layerNeurons); // add the current layer's neuron weights
				deltaWeights.add(deltaLayerNeurons); // add the current layer's previous neuron weights
			}

			network.add(layer);
		}
	}

	public void setTargets (char[] targets) {
		this.targets = targets;

		if (targets.length != network.get(network.size() - 1).size())
			hardMatch = false;
	}

	public void init() {
		epochCounter = 0;
		numCorrect = 0;
	}

	public void feedForward (ArrayList<Double> distortedImage) {
		char letter = (char)Math.floor(distortedImage.get(0));

		for (int i = 0; i < network.get(0).size() - 1; i++) { // set all the input values to the corresponding neurons except the last -1 neuron
			network.get(0).set(i, distortedImage.get(i + 1));
		}

		for (int i = 1; i < network.size(); i++) { // iterate through all the layers except the input layer
			for (int j = 0; j < network.get(i).size() - 1; j++) { // iterate through all the neurons in current layer except the -1 neuron at the bottom
				network.get(i).set(j, activation(i, j)); // set the value of neuron j at level i

				if (i == network.size() - 1){
					network.get(i).set(j + 1, activation(i, j + 1)); // set the value of neuron j at level i
				}

			}
		}

		int matchingIndex = getMatchingIndex(letter);
		boolean matches = true;

		for (int i = 0; i < network.get(network.size() - 1).size(); i++) { // iterate all output neurons
			if (i == matchingIndex) {
				if (network.get(network.size() - 1).get(i) != 1.0) {
					matches = false;
					break;
				}
			}
			else {
				if (network.get(network.size() - 1).get(i) != 0.0) {
					matches = false;
					break;
				}
			}
		}

		if (matches) {
			System.out.println("MATCHED LETTER '" + letter + "' at index " + matchingIndex);
			numCorrect++; // letter was correctly and uniformly identified
		}

		epochCounter++;
	}

	private int getMatchingIndex(char letter) {
		int matchingIndex;

		if (!hardMatch) {
			matchingIndex = network.get(network.size() - 1).size() - 1; // the output neuron that must be 1, all other must be 0

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

	private double activation (int layer, int neuron) {
		double netVal = 0.0;

		for (int i = 0; i < network.get(layer - 1).size(); i++) { // iterate all the weights i of neuron at layer
			netVal += network.get(layer - 1).get(i) * weights.get(layer - 1).get(neuron).get(i);
		}

		return 1 / (1 + Math.pow(Math.E, netVal));
	}

	public void backPropagation (ArrayList<Double> distortedImage) {
		char letter = (char)Math.floor(distortedImage.get(0));
		int matchingIndex = getMatchingIndex(letter);
		int inputLayerIndex = 0;
		int hiddenLayerIndex = 1;
		int outputLayerIndex = 2;
		ArrayList<Double> outputErrorSignals = new ArrayList<>();

		for (int i = 0; i < network.get(outputLayerIndex).size(); i++) { // iterate all output neurons
			double target = 0.0; // default target for a neuron is 0.0
			double neuronValue = network.get(outputLayerIndex).get(i); // value of current neuron

			if (i == matchingIndex)
				target = 1.0; // if this neuron is the one that has to be true then set the target to 1.0

			double errorSignal = -1.0 * (target - neuronValue) * (1 - neuronValue) * neuronValue; // calculate the error signal of the current output neuron
			outputErrorSignals.add(errorSignal);
			ArrayList<Double> newDeltaWeights = new ArrayList<>();
			ArrayList<Double> currNeuronWeights = weights.get(hiddenLayerIndex).get(i);

			for (int j = 0; j < currNeuronWeights.size(); j++) { // iterate all the weights leading to neuron i
				newDeltaWeights.add(-1.0 * learningRate * errorSignal * network.get(hiddenLayerIndex).get(j)); // calculate new delta weight
				currNeuronWeights.set(j, currNeuronWeights.get(j) + newDeltaWeights.get(j) + momentum * deltaWeights.get(hiddenLayerIndex).get(i).get(j)); // add current weight and new delta weight plus momentum of previous delta weight
				deltaWeights.get(hiddenLayerIndex).get(i).set(j, newDeltaWeights.get(j)); // set the old delta weight to new delta weight
			}
		}

		for (int i = 0; i < network.get(hiddenLayerIndex).size(); i++) { // iterate all neurons in hidden layer
			double neuronValue = network.get(hiddenLayerIndex).get(i); // value of current neuron

			double errorSignal = 0.0; // calculate the error signal of the current hidden neuron
			ArrayList<Double> newDeltaWeights = new ArrayList<>(weights.get(inputLayerIndex).get(i).size());
			ArrayList<Double> currNeuronWeights = weights.get(inputLayerIndex).get(i);

			for (int j = 0; j < network.get(outputLayerIndex).size(); j++) { // iterate all output neurons
				errorSignal += outputErrorSignals.get(j) * currNeuronWeights.get(j) * (1 - neuronValue) * neuronValue; // calculate the current error neuron's signal
			}

			for (int j = 0; j < currNeuronWeights.size(); j++) { // iterate all the weights leading to neuron i
				newDeltaWeights.add(-1.0 * learningRate * errorSignal * network.get(inputLayerIndex).get(j)); // calculate new delta weight
				currNeuronWeights.set(j, currNeuronWeights.get(j) + newDeltaWeights.get(j) + momentum * deltaWeights.get(inputLayerIndex).get(i).get(j)); // add current weight and new delta weight plus momentum of previous delta weight
				deltaWeights.get(inputLayerIndex).get(i).set(j, newDeltaWeights.get(j)); // set the old delta weight to new delta weight
			}
		}
	}
}
