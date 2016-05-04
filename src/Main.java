import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Random;
import java.nio.file.Files;
import java.nio.file.FileSystems;

public class Main {
	private static final int MAX_EPOCH = 10000;

    public static void main(String[] args) {
		ArrayList<ArrayList<Double>> distortedImages = new ArrayList<>();
		FileInputStream fis = null;

		try {
			fis = new FileInputStream("letter-recognition.data");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		BufferedReader br = new BufferedReader(new InputStreamReader(fis));
		int min = 0;
		int max = 15;
        String line;

        try {
			System.out.println("Reading data...");

            while ((line = br.readLine()) != null) {
				ArrayList<Double> distortedImage = new ArrayList<>();

				String[] split = line.split(",");
				distortedImage.add((double)split[0].charAt(0));

				for (int i = 1; i < split.length; i++)
					distortedImage.add((double) Integer.parseInt(split[i]));

				distortedImages.add(distortedImage);
			}
        } catch (IOException e) {
            e.printStackTrace();
        }

		try {
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

		int[] topology = {distortedImages.get(0).size() - 1, (distortedImages.get(0).size() - 1) * 3, 26};
		NeuralNetwork nn = new NeuralNetwork(topology);
//		char[] targets = {'A', 'E', 'O', 'U', 'I'}; // , 'E', 'O', 'U', 'I'
//		nn.setTargets(targets);

		normalize(distortedImages, min, max);
		boolean stop = false;
		double avgMSE = 0.0;

		try {
			Files.delete(FileSystems.getDefault().getPath("NNStats.txt"));
		} catch (Exception ignored) {}

		for (int i = 0; i < MAX_EPOCH && !stop; i++) {
			System.out.println("Epoch " + (i + 1));
			shuffle(distortedImages);
			nn.init();

			for (int j = 0; j < Math.ceil(distortedImages.size() * 0.6) && !stop; j++) {
				nn.feedForward(distortedImages.get(j));
				nn.backPropagation(distortedImages.get(j));
			}

			nn.adjustments();

			double trainCorrect = nn.getNumCorrect() / (distortedImages.size() * 0.6);

			nn.init();

			for (int j = (int)Math.floor(distortedImages.size() * 0.6); j < Math.ceil(distortedImages.size() * 0.8) && !stop; j++) {
				nn.feedForward(distortedImages.get(j));
			}

			double generalCorrect = nn.getNumCorrect() / (distortedImages.size() * 0.2);
			nn.log(i + 1, trainCorrect, generalCorrect);

			if (generalCorrect >= nn.getTrainingAccuracy())
				break;

			nn.init();

			for (int j = (int)Math.floor(distortedImages.size() * 0.8); j < distortedImages.size() && !stop; j++) {
				nn.feedForward(distortedImages.get(j));
			}

			avgMSE = (avgMSE * i + nn.getMeanSquareError()) / (i + 1);
			double variance = nn.getMeanSquareError() / (distortedImages.size() * 0.2);
			double stdDeviation = Math.sqrt(variance);

			System.out.println(avgMSE + "  " + stdDeviation + "  " + nn.getMeanSquareError());

			if (avgMSE + stdDeviation < nn.getMeanSquareError())
				break;
		}
	}

	public static void normalize(ArrayList<ArrayList<Double>> distortedImages, int min, int max) {
		System.out.println("Normalizing data...");

		for (int i = 0; i < distortedImages.size(); i++) { // iterate all images
			for (int j = 1; j < distortedImages.get(i).size(); j++) { // iterate current image size, skip the first element which is the character
				double tmp = distortedImages.get(i).get(j);
				tmp = (tmp - min) / (max - min);
				tmp = tmp * (2 * Math.sqrt(3)) - Math.sqrt(3);
				distortedImages.get(i).set(j, tmp);
			}
		}
	}

	public static void shuffle(ArrayList<ArrayList<Double>> distortedImages) {
		//System.out.println("\tShuffling data...");
		Random random = new Random();

		for (int i = 0; i < distortedImages.size(); i++) {
			int index = (int)Math.round(random.nextDouble() * (distortedImages.size() - 1));
			ArrayList<Double> tmp = distortedImages.get(i);
			distortedImages.set(i, distortedImages.get(index));
			distortedImages.set(index, tmp);
		}
	}
}
