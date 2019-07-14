using System;

namespace mnist
{
    class Program
    {
        static void Main(string[] args)
        {
            void print_vector(double[] v){
                for (int i = 0; i < v.Length; i++)
                {
                    Console.WriteLine(v[i]);
                }
            }
            
            int nr_of_layers = 3;
            int[] nr_of_nodes = {2,5,4,3};
            double learning_rate = 0.1;

            neuralNetwork test = new neuralNetwork(nr_of_layers,nr_of_nodes,learning_rate);
            //test.print_weights();
            double[][] x = new double[][]{
                new double[] {1,2},
                new double[] {1,2}
            }; 
            double[][] y = {
                new double[] {1,2,3},
                new double[] {1,2,3}
            };
            
            double[][] data = new double[3][];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = new double[4];
            }

            //double[,] inputs = {{0, 1},
            //                    {1, 0},
            //                    {0, 0},
            //                    {1, 1}};
            //double[,] outputs = {{1},
            //                    {1},
            //                    {0},
            //                    {0}};
            //test.Train(x,y);
            print_vector(test.FeedForward(x[0]));
            
        }
    }
}
