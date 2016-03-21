/* For a set of training data, this file does two things:
 * 1) Prints each of the features in descending order of covariance.
 * 		Note: Also writes to a text file to be used in PcaKnn.java feature test
 * 2) Prints each of the features in descending order of eigenvalues (variance of principal components).
 *
 * @author Scott Weaver
 */
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

import javafx.util.Pair;

public class PCA {

	private static String FEATURE_FILE_PATH = "Data/TrainingData/training_data.csv";
	private static String FEATURE_LIST_PATH = "Data/feature_list.txt";
	
	public static void main(String[] args) {
		
		System.out.println("Feature File Path " + FEATURE_FILE_PATH);
		
		ArrayList<double[]> featureData = new ArrayList<>();
		ArrayList<String> featureClassification = new ArrayList<>();
		
		if ((new File(FEATURE_FILE_PATH)).isFile()) {
			String columnHeaders = readFeatureFile(FEATURE_FILE_PATH, featureData, featureClassification);
			ArrayList<String> features = getColumnHeaders(columnHeaders);
			
			performPCA(featureData, features);
		} else {
			System.err.println("Feature File does not exist.");
		}
	}
	
	private static void performPCA(ArrayList<double[]> featureData, ArrayList<String> features) {
		int numFeatures = featureData.get(0).length;
		double dataAverage[] = new double[numFeatures];
		
		getAverage(featureData, dataAverage);
    	
		double[][] covarianceMatrix = PCAUtilities.getCovarianceMatrix(featureData, dataAverage, numFeatures);
    	ArrayList<Double> covarianceList = getPrincipalDiagonal(covarianceMatrix, numFeatures);
    	ArrayList<Pair<Double,String>> covarianceFeatures = getFeatureCovariance(covarianceList, features);

    	System.out.println();
	    System.out.println("------------------------------");
	    System.out.println("Covariance");
	    try {
	    	BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(FEATURE_LIST_PATH, true));
	    	for (Pair<Double,String> covariance : covarianceFeatures) {
	    		System.out.println("Feature " + covariance.getValue() + ":\t\t" + covariance.getKey());
	    		bufferedWriter.write(covariance.getValue());
	    		bufferedWriter.newLine();
	    	}
	    	bufferedWriter.close();
	    } catch (IOException e) {
	    	System.err.println("Unable to print ordered feature covariance list.");
	    }
	    
	    ArrayList<Double> eigenvalueList = PCAUtilities.getEigenvalueMatrix(covarianceMatrix, numFeatures);
	    
	    System.out.println();
	    System.out.println("------------------------------");
	    System.out.println("Eigenvalues");
	    double totalSum = 0;
	    for (Double eigenvalue : eigenvalueList) {
	    	totalSum += eigenvalue;
	    }
	    int count = 0;
	    double cumSum = 0;
	    for (Double eigenvalue : eigenvalueList) {
	    	cumSum += eigenvalue;
	    	System.out.println("Eigenvalue " + count + ":\t" + eigenvalue + ",\t\tPercentage of sum: " + (cumSum / totalSum));
	    	count++;
	    }
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
	
	private static ArrayList<Double> getPrincipalDiagonal(double[][] matrix, int numFeatures) {
		ArrayList<Double> principalDiagonal = new ArrayList<>(numFeatures);
		
		for (int i = 0; i < numFeatures; i++) {
			principalDiagonal.add(Double.valueOf(matrix[i][i]));
		}
		
		return principalDiagonal;
	}
	
	private static ArrayList<Pair<Double,String>> getFeatureCovariance(ArrayList<Double> covarianceList, ArrayList<String> features) {
		ArrayList<Pair<Double,String>> covarianceFeatures = new ArrayList<>();
		
		int index = 0;
		for (Double covariance : covarianceList) {
			covarianceFeatures.add(new Pair<Double,String>(covariance, features.get(index)));
			index++;
		}
		
        //sort the eigenvalues in descending order.
        //the largest eigenvalue corresponds with the most significant (highest variance) principal component.
        Collections.sort(covarianceFeatures, new Comparator<Pair<Double,String>>() {
            @Override
            public int compare(Pair<Double,String> cv1, Pair<Double,String> cv2) {
                double covariance1 = cv1.getKey().doubleValue();
                double covariance2 = cv2.getKey().doubleValue();
                return (covariance1 == covariance2) ? 0 : (covariance1 < covariance2 ? 1 : -1);
            }
        });
		
		return covarianceFeatures;
	}
	
    private static ArrayList<String> getColumnHeaders(String line) {
    	ArrayList<String> columnHeaders = new ArrayList<>();
    	
    	String dataCompsStr[] = line.substring(line.indexOf(",") + 1).split(",");
    	
    	for (String feature : dataCompsStr) {
    		columnHeaders.add(feature);
    	}
    	
    	return columnHeaders;
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
