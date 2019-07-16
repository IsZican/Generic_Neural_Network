using System;
using System.Diagnostics;

namespace generic
{
    class Program
    {
        static void Main(string[] args)
        {
            Stopwatch stopwatch = Stopwatch.StartNew();
            
            int nr_of_layers = 4;
            int[] nr_of_nodes = {2,5,4,1};
            double learning_rate = 0.1;

            neuralNetwork test = new neuralNetwork(nr_of_layers,nr_of_nodes,learning_rate);
            //test.print_weights();
            double[][] x = new double[][]{
                new double[] {1,0},
                new double[] {1,1},
                new double[] {0,1},
                new double[] {0,0}
            }; 
            double[][] y = {
                new double[] {1},
                new double[] {0},
                new double[] {1},
                new double[] {0},
            };
            
            test.Train(x, y);
            //test.FeedForward(x[0]).print_data();
            
            stopwatch.Stop();
            TimeSpan ts = stopwatch.Elapsed;
            string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
            ts.Hours, ts.Minutes, ts.Seconds,
            ts.Milliseconds );
            Console.WriteLine("RunTime " + elapsedTime);
            //Console.WriteLine();
            //Console.WriteLine(stopwatch.ElapsedMilliseconds);
        }
    }
}
