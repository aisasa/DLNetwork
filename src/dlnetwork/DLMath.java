package dlnetwork;

import java.util.*;

public class DLMath {
    
    public static double[][] dotProd(double[][] a, double[][] b){
        int nRowsA = a.length;
        int nColsA = a[0].length;
        int nRowsB = b.length;
        int nColsB = b[0].length;
        if(nColsA != nRowsB) throw new RuntimeException("No matching matrices' dimensions");
        double[][] dp = new double[a.length][b[0].length];
        for(int i=0; i<nRowsA; i++)
            for(int j=0; j<nColsB; j++)
                for(int k=0; k<nColsA; k++)
                    dp[i][j] += a[i][k]*b[k][j];
        return dp;
    }
    
    public static double[] costQuadDeriv(double y[], double real[]){
        double[] cd = new double[y.length];
        for(int i=0; i<y.length; i++){
            cd[i] = y[i]-real[i];  
        }
        return cd;
    }
    
    public static double[] costCrossEntrDeriv(double y[], double real[]){
        double[] cd = new double[y.length];
        for(int i=0; i<y.length; i++){
            cd[i] = y[i]-real[i];  
        }
        return cd;
    }
    
    public static double[][] sigmoid(double[][] z){
        double[][] sigm = new double[z.length][z[0].length];
        //double[] sigm = Arrays.copyOf(z, z.length);
        for(int i=0; i<z.length; i++)
            for(int j=0; j<z[0].length; j++)
                sigm[i][j] = 1.0/(1.0 + Math.exp(-z[i][j])); 
        return sigm;
    }
    
    public static double[][] sigmoidDeriv(double[][] z){
        double[][] sigm_d = sigmoid(z);
        for(int i=0; i<z.length; i++)
            for(int j=0; j<z[0].length; j++)
                sigm_d[i][j] = sigm_d[i][j]*(1-sigm_d[i][j]);
        return sigm_d;
    }
    
    public static double[] vTranspose(double[][] v){
        double[] vT = new double[v.length];
        for(int i=0; i<v.length; i++)
            vT[i] = v[i][0]; 
        return vT;
    }
    
    public static double[][] mTranspose(double[][] m){
        double[][] mT = new double[m[0].length][m.length];
        for(int i=0; i<m[0].length; i++)
            for(int j=0; j<m.length; j++)
                mT[i][j] = m[j][i]; 
        return mT;
    }
    
    public static ArrayList<double[][]> addMatrInLists(ArrayList<double[][]> a, ArrayList<double[][]> b){
        ArrayList<double[][]> listSum = new ArrayList(a.size());
        double[][] sum; 
        for(int i=0; i<a.size(); i++){
            sum = new double[a.get(i).length][a.get(i)[0].length];
            for(int j=0; j<a.get(i).length; j++)
                for(int k=0; k<a.get(i)[j].length; k++)
                    sum[j][k] = a.get(i)[j][k] + b.get(i)[j][k];
            listSum.add(sum);
        }
        return listSum;        
    }
    
    public static double[] getOutputVector(int output){
        double[] ov = new double[10];
        ov[output] = 1.0;
        return ov;
    }
    
    public static void shuffle(double[][] a, double[] b){
        // A basic Fisher-Yates shuffle
        Random rand = new Random();
        for(int i=a.length-1; i>0; --i){
            int j = rand.nextInt(i+1);
            double[] tempA = a[i];
            double tempB = b[i];
            a[i] = a[j];
            b[i] = b[j];
            a[j] = tempA;
            b[j] = tempB;
        }
    }
}
