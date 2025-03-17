import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

public class MatrixMultiplication {
    public static void main(String[] args) throws IOException {
        if (args.length != 2) {
            System.out.println("Usage: java MatrixMultiplication file1.csv file2.csv");
            return;
        }

        String file1 = args[0];
        String file2 = args[1];

        double [][] matrix1, matrix2;
        try {
            matrix1 = readMatrixFromFile(file1);
            matrix2 = readMatrixFromFile(file2);
        }
        catch(Exception exp)
        {
            System.out.println(exp.getMessage());
            return;
        }

        int rows1 = matrix1.length;
        int cols1 = matrix1[0].length;
        int rows2 = matrix2.length;
        int cols2 = matrix2[0].length;

        if (cols1 != rows2) {
            System.out.println("Error: matrices cannot be multiplied");
            return;
        }

        double[][] result = new double[rows1][cols2];

        // TODO: Write your matrix multiplication code here 
        
        printNumpyMatrix(result);
    }

    private static double[][] readMatrixFromFile(String filename) throws IOException, Exception {
        BufferedReader reader = new BufferedReader(new FileReader(filename));

        String line = reader.readLine();
        String[] parts = line.split(",");
        int cols = parts.length;

        double[][] matrix = new double[10][cols];
        int rows = 0;

        do {
            if (rows == matrix.length) {
                matrix = Arrays.copyOf(matrix, matrix.length * 2);
            }

            for (int i = 0; i < cols; i++) {
                matrix[rows][i] = Double.parseDouble(parts[i]);
            }

            rows++;

            line = reader.readLine();
            if (line != null) {
                parts = line.split(",");
				
				// TODO: Check column size consistency
				/*
                if () {
                    String message = "Error! Column size is not the same in " + filename + " at line " + (rows+1);
                    throw new Exception(message);
                }*/
            }
        } while (line != null);

        return Arrays.copyOf(matrix, rows);
    }

    private static void printMatrix(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                System.out.print(matrix[i][j] + " ");
            }

            System.out.println();
        }
    }

    private static void printNumpyMatrix(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;

        System.out.print("[");
        for (int i = 0; i < rows; i++) {
            System.out.print("[");
            for (int j = 0; j < cols; j++) {
                System.out.print(matrix[i][j]);
                if (j < cols - 1) {
                    System.out.print(",");
                }
            }
            System.out.print("]");
            if (i < rows - 1) {
                System.out.println(",");
            } else {
                System.out.println("]");
            }
        }
    }
}
