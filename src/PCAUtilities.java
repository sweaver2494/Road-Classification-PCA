/*
 *
 * @author Scott Weaver
 */
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;

public class PCAUtilities {
	public static double[][] getCovarianceMatrix(ArrayList<double[]> featureData, double[] dataAvg, int numFeatures) {

    	// create a copy of featureData
        ArrayList<double[]> featureDataAdjust = new ArrayList<>();
        for (double[] src : featureData) {
        	double[] dest = new double[numFeatures];
        	System.arraycopy( src, 0, dest, 0, src.length );
        	
        	featureDataAdjust.add(dest);
        }
        
        //creating data adjust
        for (double dataAdjustComps[] : featureDataAdjust) {
            for (int i = 0; i < numFeatures; i++) {
                dataAdjustComps[i] -= dataAvg[i];
            }
        }
        
        double[][] covarianceMatrix = new double[numFeatures][numFeatures];

        for (int i = 0; i < numFeatures; i++) {
            for (int j = 0; j < numFeatures; j++) {
                covarianceMatrix[i][j] = calculateCovariance(featureDataAdjust, i, j);
            }
        }
        
        return covarianceMatrix;
		
	}
	
	public static ArrayList<Double> getEigenvalueMatrix(double[][] covarianceMatrix, int numFeatures) {
		
        EigenvalueDecomposition evd = new EigenvalueDecomposition(new Matrix(covarianceMatrix));

        //eigenValues contains the eigenvalues of the covariance matrix
        double[] eigenValues = evd.getRealEigenvalues();
        
        //create a n ArrayList copy
        ArrayList<Double> eigenValuesList = new ArrayList<>(numFeatures);
        for (int i = 0; i < numFeatures; i++) {
            eigenValuesList.add(eigenValues[i]);
        }
        //sort the eigenvalues in descending order.
        //the largest eigenvalue corresponds with the most significant (highest variance) principal component.
        Collections.sort(eigenValuesList, new Comparator<Double>() {
            @Override
            public int compare(Double ev1, Double ev2) {
                double eigenVal1 = ev1.doubleValue();
                double eigenVal2 = ev2.doubleValue();
                return (eigenVal1 == eigenVal2) ? 0 : (eigenVal1 < eigenVal2 ? 1 : -1);
            }
        });
        
        return eigenValuesList;
	}
	
    private static double calculateCovariance(ArrayList<double[]> fullDataAdjust, int i, int j) {

        double metricAdjustProdTotal = 0.0;  // the final numerator in the covariance formula i.e Summation[(Xi-Xmean)*(Yi-Ymean)]

        for (double dataAdjustComps[] : fullDataAdjust) {
            metricAdjustProdTotal += dataAdjustComps[i] * dataAdjustComps[j];
        }
        return metricAdjustProdTotal / (fullDataAdjust.size() - 1);
    }
}
