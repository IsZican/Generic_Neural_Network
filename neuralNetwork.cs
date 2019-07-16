using System;


namespace generic{
    class neuralNetwork{
        
        #region variables
        int nr_of_layers;
        int[] nr_of_nodes;
        double learning_rate;        
        matrix[] weights;
        matrix[] bias;
        matrix[] z;
        matrix[] a;

        #endregion

        #region constructors
        /// <summary>
        /// <param name="nr_of_layers">The number of layers the neural network will have.</param>
        /// <param name="nr_of_nodes">A list containing the number of nodes for every layer.</param>
        /// <param name="learning_rate"></param>
        /// </summary>
        public neuralNetwork(int nr_of_layers, int[] nr_of_nodes, double learning_rate){
            this.nr_of_layers = nr_of_layers;
            this.nr_of_nodes = nr_of_nodes;
            this.learning_rate = learning_rate;
            this.z = new matrix[nr_of_layers - 1];
            this.a = new matrix[nr_of_layers];
            this.weights = new matrix[nr_of_layers - 1];
            this.bias = new matrix[nr_of_layers -1];
            for (int i = 0; i < nr_of_layers - 1; i++)
            {
                this.weights[i] = new matrix(nr_of_nodes[i], nr_of_nodes[i+1]);
                this.weights[i].randomize_data();
                this.bias[i] = new matrix(1,nr_of_nodes[i+1]);
                this.bias[i].randomize_data();
            }
        }

        #endregion

        #region Methods
        public void print_weights(){
            for (int i = 0; i < nr_of_layers - 1; i++)
            {
                weights[i].print_data();
                Console.WriteLine();
            }
        }

        public void print_biases(){
            for (int i = 0; i < nr_of_layers - 1; i++)
            {
                bias[i].print_data();
                Console.WriteLine();
            }
        }

        /// <summary>
        /// Predict the value of each output node using trained weights of a neural network.
        /// </summary>
        /// <param name="input"> The input layer vector.</param>
        /// <returns>A vector of size (number of nodes in the output layer), containing the results. </returns>
        public matrix FeedForward(double[][] input){
            matrix result = new matrix(input.Length, nr_of_nodes[nr_of_nodes.Length - 1]);
            for (int i = 0; i < input.Length; i++)
            {
                matrix z_ = new matrix(input[i]);
                a[0] = new matrix(input[i]);
                for (int j = 0; j < this.nr_of_layers - 1; j++)
                {
                    z_ = weights[j].Transpose * a[j] + bias[j].Transpose;
                    a[j+1] = z_;
                    a[j+1].ApplySigmoid();
                }
                //z[L]
                //a[L]
                for (int j = 0; j < result.columns; j++)
                {
                    result.data[i][j] = a[nr_of_layers - 1].data[j][0];
                }
            }
            return result;
        }

        /// <summary>
        /// Predict the value of each output node using trained weights of a neural network.
        /// </summary>
        /// <param name="input"> The input layer vector.</param>
        /// <returns>A vector of size (number of nodes in the output layer), containing the results. </returns>
        public matrix FeedForward(double[] input){
            a[0] = new matrix(input);
            for (int j = 0; j < this.nr_of_layers - 1; j++)
            {
                z[j] = weights[j].Transpose * a[j] + bias[j].Transpose;
                a[j+1] = z[j];
                a[j+1].ApplySigmoid();
            }
            /*
            TODO: to add different activation function for last layer.
            z[L]
            a[L]
            */
            return a[nr_of_layers -1];
        }

        double CostFunction_L(int layer, int node, matrix target){
            return (a[layer].data[node][0] * (1 - a[layer].data[node][0]) * (a[layer].data[node][0] - target.data[node][0]));
        }
        double CostFunction(int layer, int node, double[][] prev_cost, int prev_nodes){
            double error = (a[layer].data[node][0] * (1 - a[layer].data[node][0]));
            double sum = 0;

            for (int i = 0; i < prev_nodes; i++)
            {
                sum += prev_cost[layer + 1][i] * weights[layer].data[node][i];
            }
            return error * sum;
        }

        public void Train(double[][] input, double[][] target, int epochs = 5){
            double[][] cost = new double[nr_of_layers][];
            double[][] gradients = new double[nr_of_layers][];
            for (int i = 0; i < nr_of_layers; i++)
            {
                cost[i] = new double[nr_of_nodes[i]];
                gradients[i] = new double[nr_of_nodes[i]];
            }
            matrix yHat;
            matrix y;

            for (int i = 0; i < epochs; i++)
            {
                for (int j = 0; j < input.Length; j++)
                {
                    yHat = FeedForward(input[j]);
                    y = new matrix(target[j]);
                    
                    // layers 0,1,2; a has +1
                    for (int n = 0; n < nr_of_nodes[nr_of_layers - 1]; n++)
                    {
                        cost[nr_of_layers - 1][n] = this.CostFunction_L(nr_of_layers - 1, n, y);//2
                    }

                    for (int l = nr_of_layers - 2; l >= 0; l--)//1
                    {
                        for (int n = 0; n < nr_of_nodes[l]; n++)
                        {
                            cost[l][n] = this.CostFunction(l, n, cost, nr_of_nodes[l+1]);
                            //gradients[l][n] =                             
                        }
                    }
                }
                Console.WriteLine("epoch " + i + " done.");
            }
            for (int i = 0; i < cost.Length; i++)
            {
                for (int j = 0; j < cost[i].Length; j++)
                {
                    Console.Write(cost[i][j] + " ");
                }
                Console.WriteLine();
            }
        }

        #endregion
    }
}