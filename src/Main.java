import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Random;

public class Main {
	private static final int NUM_SESSIONS = 30;

    public static void main(String[] args) {
		ArrayList<ArrayList<Double>> distortedImages = new ArrayList<>();
		FileInputStream fis = null;

		try {
			fis = new FileInputStream("letter-recognition.data");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		BufferedReader br = new BufferedReader(new InputStreamReader(fis));
		int[] min = null;
		int[] max = null;
        String line;

        try {
			System.out.println("Reading data...");

            while ((line = br.readLine()) != null) {
				ArrayList<Double> distortedImage = new ArrayList<>();

				String[] split = line.split(",");
				distortedImage.add((double)split[0].charAt(0));

				if (min == null){
					min = new int[split.length - 1];
					max = new int[split.length - 1];

					for (int i = 0; i < min.length; i++) {
						min[i] = 0;
						max[i] = 0;
					}
				}

				for (int i = 1; i < split.length; i++)
					distortedImage.add((double) Integer.parseInt(split[i]));

				for (int i = 1; i < distortedImage.size(); i++) {
					if (distortedImage.size() < min[i - 1])
						min[i - 1] = (int) Math.floor(distortedImage.get(i));

					if (distortedImage.size() > max[i - 1])
						max[i - 1] = (int) Math.floor(distortedImage.get(i));
				}

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

		int[] topology = {distortedImages.get(0).size() - 1, distortedImages.get(0).size() - 1, 2};
		NeuralNetwork nn = new NeuralNetwork(topology);
		char[] targets = {'A'};
		nn.setTargets(targets);

		normalize(distortedImages, min, max);
		boolean stop = false;

		for (int i = 0; i < NUM_SESSIONS && !stop; i++) {
			System.out.println("Session " + (i + 1));
			shuffle(distortedImages);
			nn.init();

			for (int j = 0; j < distortedImages.size() * 0.6 && !stop; j++) {
				nn.feedForward(distortedImages.get(j));
				nn.backPropagation(distortedImages.get(j));
			}

//			if (i == 0)
//				stop = true;
		}
	}

	public static void normalize(ArrayList<ArrayList<Double>> distortedImages, int[] min, int[] max) {
		System.out.println("Normalizing data...");

		for (int i = 0; i < distortedImages.size(); i++) {
			for (int j = 1; j < distortedImages.get(i).size(); j++) {
				double tmp = distortedImages.get(i).get(j);
				distortedImages.get(i).set(j, (tmp - min[j - 1]) / (max[j - 1] - min[j - 1]));
			}
		}
	}

	public static void shuffle(ArrayList<ArrayList<Double>> distortedImages) {
		System.out.println("Shuffling data...");
		Random random = new Random();

		for (int i = 0; i < distortedImages.size(); i++) {
			int index = (int)Math.round(random.nextDouble() * (distortedImages.size() - 1));
			ArrayList<Double> tmp = distortedImages.get(i);
			distortedImages.set(i, distortedImages.get(index));
			distortedImages.set(index, tmp);
		}
	}
}
