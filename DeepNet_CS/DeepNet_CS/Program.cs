using System;
using System.Diagnostics;

namespace ConsoleApp1
{
    class Program
    {
        static void Main(string[] args)
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();
            Console.WriteLine("\nBegin deep net training demo \n");

            int numInput = 4;
            int[] numHidden = new int[] { 10, 10, 10 };
            int numOutput = 3;

            int numDataItems = 500;
            Console.WriteLine("Generating " + numDataItems + " artificial training data items ");
            double[][] trainData = MakeData(numDataItems, numInput, numHidden, numOutput, 5);
            Console.WriteLine("\nDone. Training data is: ");
            ShowMatrix(trainData, 3, 2, true);

            Console.WriteLine("\nCreating a 4-(10,10,10)-3 deep neural network (tanh & softmax) \n");
            var dn = new DeepNet(numInput, numHidden, numOutput);
            //dn.Dump();

            int maxEpochs = 10000;
            double learnRate = 0.001;
            double momentum = 0.01;
            Console.WriteLine("Setting maxEpochs = " + maxEpochs);
            Console.WriteLine("Setting learnRate = " + learnRate.ToString("F3"));
            Console.WriteLine("Setting momentumm = " + momentum.ToString("F3"));
            Console.WriteLine("\nStart training using back-prop with mean squared error \n");
            double[] wts = dn.Train(trainData, maxEpochs, learnRate, momentum, 10);  // show error every maxEpochs / 10 
            Console.WriteLine("Training complete \n");

            double trainError = dn.Error(trainData, false);
            double trainAcc = dn.Accuracy(trainData, false);
            Console.WriteLine("Final model MS error = " + trainError.ToString("F4"));
            Console.WriteLine("Final model accuracy = " + trainAcc.ToString("F4"));

            Console.WriteLine("\nEnd demo ");
            Console.WriteLine(sw.ElapsedMilliseconds.ToString() + "ms");
            Console.ReadLine();

        } // Main

        static double[][] MakeData(int numItems, int numInput, int[] numHidden, int numOutput, int seed)
        {
            // generate data using a Deep NN (tanh hidden activation)
            DeepNet dn = new DeepNet(numInput, numHidden, numOutput);  // make a DNN generator
            Random rrnd = new Random(seed);  // to make random weights & biases, random input vals
            double wtLo = -9.0;
            double wtHi = 9.0;
            int nw = DeepNet.NumWeights(numInput, numHidden, numOutput);
            double[] wts = new double[nw];

            for (int i = 0; i < nw; ++i)
                wts[i] = (wtHi - wtLo) * rrnd.NextDouble() + wtLo;
            dn.SetWeights(wts);

            double[][] result = new double[numItems][];  // make the result matrix holder
            for (int r = 0; r < numItems; ++r)
                result[r] = new double[numInput + numOutput];  // allocate the cols

            double inLo = -4.0;    // pseudo-Gaussian scaling
            double inHi = 4.0;
            for (int r = 0; r < numItems; ++r)  // each row
            {
                double[] inputs = new double[numInput];  // random input values

                for (int i = 0; i < numInput; ++i)
                    inputs[i] = (inHi - inLo) * rrnd.NextDouble() + inLo;

                //ShowVector(inputs, 2);

                double[] probs = dn.ComputeOutputs(inputs);  // compute the outputs (as softmax probs) like [0.10, 0.15, 0.55, 0.20]
                                                             //dn.Dump();
                                                             //Console.ReadLine();
                                                             //ShowVector(probs, 4);
                double[] outputs = ProbsToClasses(probs);  // convert to outputs like [0, 0, 1, 0]

                int c = 0;
                for (int i = 0; i < numInput; ++i)
                    result[r][c++] = inputs[i];
                for (int i = 0; i < numOutput; ++i)
                    result[r][c++] = outputs[i];
                //Console.WriteLine("");
            }
            return result;

        } // MakeData

        static double[] ProbsToClasses(double[] probs)
        {
            double[] result = new double[probs.Length];
            int idx = MaxIndex(probs);
            result[idx] = 1.0;
            return result;
        }

        static int MaxIndex(double[] probs)
        {
            int maxIdx = 0;
            double maxVal = probs[0];

            for (int i = 0; i < probs.Length; ++i)
            {
                if (probs[i] > maxVal)
                {
                    maxVal = probs[i];
                    maxIdx = i;
                }
            }
            return maxIdx;
        }

        public static void ShowMatrix(double[][] matrix, int numRows,
          int decimals, bool indices)
        {
            int len = matrix.Length.ToString().Length;
            for (int i = 0; i < numRows; ++i)
            {
                if (indices == true)
                    Console.Write("[" + i.ToString().PadLeft(len) + "]  ");
                for (int j = 0; j < matrix[i].Length; ++j)
                {
                    double v = matrix[i][j];
                    if (v >= 0.0)
                        Console.Write(" "); // '+'
                    Console.Write(v.ToString("F" + decimals) + "  ");
                }
                Console.WriteLine("");
            }

            if (numRows < matrix.Length)
            {
                Console.WriteLine(". . .");
                int lastRow = matrix.Length - 1;
                if (indices == true)
                    Console.Write("[" + lastRow.ToString().PadLeft(len) + "]  ");
                for (int j = 0; j < matrix[lastRow].Length; ++j)
                {
                    double v = matrix[lastRow][j];
                    if (v >= 0.0)
                        Console.Write(" "); // '+'
                    Console.Write(v.ToString("F" + decimals) + "  ");
                }
            }
            Console.WriteLine("\n");
        }

        //static void ShowMatrix(double[][] matrix, int numRows, int numDec)
        //{
        //    for (int r = 0; r < numRows; ++r)
        //    {
        //        for (int c = 0; c < matrix[r].Length; ++c)
        //        {
        //            if (matrix[r][c] >= 0.0) Console.Write(" ");  // '+'
        //            Console.Write(matrix[r][c].ToString("F" + numDec) + "  ");
        //        }
        //        Console.WriteLine("");
        //    }
        //    Console.WriteLine("");
        //}

        //static void ShowVector(double[] vector, int numDec)
        //{
        //    for (int i = 0; i < vector.Length; ++i)
        //    {
        //        if (vector[i] >= 0.0) Console.Write(" ");
        //        Console.Write(vector[i].ToString("F" + numDec) + "  ");
        //    }
        //    Console.WriteLine("");
        //}

    }
}
