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
            int[] nr_of_nodes = {2,10,1};
            double learning_rate = 0.1;

            neuralNetwork test = new neuralNetwork(nr_of_layers,nr_of_nodes,learning_rate);
            double[] x = {1,7};
            double[,] inputs = {{0, 1},
                                {1, 0},
                                {0, 0},
                                {1, 1}};
            double[,] outputs = {{1},
                                {1},
                                {0},
                                {0}};
            print_vector(test.FeedForward(x));
            
        }
    }
}
