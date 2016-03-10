/*
 *
 * @author Scott Weaver
 */
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import javafx.util.Pair;

public class PCA {

	private static String FEATURE_FILE_PATH = "Data/FeatureFiles/features_0.csv";
	//REDUCED_LENGTH is the desired number of principal components
	private static int REDUCED_LENGTH = 10;
	
	public static void main(String[] args) {
		
		System.out.println("Feature File Path " + FEATURE_FILE_PATH);
		
		ArrayList<double[]> featureData = new ArrayList<>();
		ArrayList<String> featureClassification = new ArrayList<>();
		
		if ((new File(FEATURE_FILE_PATH)).isFile()) {
			String columnHeaders = readFeatureFile(FEATURE_FILE_PATH, featureData, featureClassification);
			
			performPCA(featureData, featureClassification);
		} else {
			System.err.println("Feature File does not exist.");
		}
	}
	
	private static void performPCA(ArrayList<double[]> featureData, ArrayList<String> featureClassification) {
		int dataSize = featureData.size();
		int numFeatures = featureData.get(0).length;
		double dataAverage[] = new double[numFeatures];
		
		getAverage(featureData, dataAverage);
		
    	ArrayList<Pair<Integer, Double>> covarianceList = new ArrayList<>();
	    
	    ArrayList<double[]> newData = PCAUtilities.performPCA(featureData, dataAverage, reducedLength, covarianceList);
	    
	    System.out.println();
	    for (Pair<Integer, Double> item : covarianceList) {
	    System.out.print(columnHeaders.get(item.getKey()) + "(" + item.getValue() + "), ");
	    }
	    System.out.println();
	}
	
	//dataAverage will contain the average value for each feature (column)
	private static void getAverage(ArrayList<double[]> featureData, double[] dataAverage) {
		int dataSize = featureData.size();
		int numFeatures = dataAverage.length;
		
		for (double[] data : featureData) {
			for (int i = 0; i < numFeatures; i++) {
				dataAverage[i] += data[i];
			}
		}
		
		for (int i = 0; i < numFeatures; i++) {
			dataAverage[i] /= dataSize;
		}
	}
	
	private static String readFeatureFile(String featureFilePath, ArrayList<double[]> featureData, ArrayList<String> featureClassification) {
		String columnHeaders = "";
		
		try {

	        BufferedReader bufferedReader = new BufferedReader(new FileReader(featureFilePath));
	        
	        columnHeaders = bufferedReader.readLine();

	        String line = bufferedReader.readLine();
	        int dataSize = line.length() - line.replace(",", "").length();
	
	        double dataAvg[] = new double[dataSize];
	
	        while (line != null) {
	        	String classification = line.substring(0, line.indexOf(","));
	            String dataCompsStr[] = line.substring(line.indexOf(",") + 1).split(",");
	
	            double dataComps[] = new double[dataSize];
	
	            for (int i = 0; i < dataSize; i++) {
	                dataComps[i] = Double.parseDouble(dataCompsStr[i]);
	                dataAvg[i] += dataComps[i];
	            }
	
	            featureData.add(dataComps);
	            featureClassification.add(classification);
	            line = bufferedReader.readLine();
	        }
	        bufferedReader.close();
	        
        } catch(IOException e) {
        	System.err.println("Cannot read feature file.");
        }
		
		return columnHeaders;
	}	
}
