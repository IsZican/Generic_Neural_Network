using System;


namespace mnist{
    class neuralNetwork{
        
        #region variables
        int nr_of_layers;
        int[] nr_of_nodes;
        double learning_rate;
        matrix[] z;
        matrix[] a;
        matrix[] weights;
        matrix[] bias;

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
            this.a = new matrix[nr_of_layers - 1];
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
        public double[] FeedForward(double[] input){
            matrix z_ = new matrix(input);
            double[] result = new double[nr_of_nodes[nr_of_nodes.Length - 1]];
            for (int i = 0; i < this.nr_of_layers - 1; i++)
            {
                z_ = z_ * weights[i];
                this.z[i] = z_;
                z_.ApplySigmoid();
                this.a[i] = z_; 
            }
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = z_.data[0,i];
            }
            return result;
        }

        /// <summary>
        /// Calculate the cost function.
        /// </summary>
        /// <param name="x"> The input/s for the neural network.</param>
        /// <param name="y"> The output/s for the neural network.</param>
        /// <returns> </returns>
        /// <remarks> yHat is the estimated output.</remarks>
        double CostFunction(double[] x, double[] y){
            double[] yHat = FeedForward(x);
            double j;
            double sum = 0;
            
            for (int i = 0; i < y.Length; i++)
            {
                sum += Math.Pow((y[i] - yHat[i]),2);
            }

            j = learning_rate * sum;
            return j;
        }

        /// <summary>
        /// Subtract 2 vectors and negate the value.
        /// Used in CostFunctionPrime.
        /// </summary>
        /// <param name="y"> First vector.</param>
        /// <param name="yHat"> Second vector.</param>    
        /// <returns> The result vector</returns>
        double[] vector_subtraction(double[] y,double[] yHat){
            for (int i = 0; i < y.Length; i++)
            {
                y[i] = -(y[i] - yHat[i]);
            }
            return y;
        }

        /// <summary>
        /// Calculate the cost function prime.
        /// </summary>
        /// <param name="x"> The input/s for the neural network.</param>
        /// <param name="y"> The output/s for the neural network.</param>
        /// <returns> </returns>
        matrix[] CostFunctionPrime(double[] x, double[] y){
            double[] yHat = FeedForward(x);
            matrix input = new matrix(x);
            matrix output = new matrix(vector_subtraction(y,yHat));
            matrix[] djdw = new matrix[nr_of_layers - 1];
            matrix[] delta = new matrix[nr_of_layers - 1];
            for (int i = nr_of_layers - 2; i >= 0; i--)
            {
                this.z[i].ApplySigmoidPrime();
                if (i == nr_of_layers - 2)
                {
                    delta[i] = output * this.z[i];
                    djdw[i] = this.a[i].Transpose * delta[i];
                }
                else if(i > 0)
                {
                    delta[i] = delta[i+1] * this.weights[i+1].Transpose * this.z[i];
                    djdw[i] = this.a[i].Transpose * delta[i];
                }
                else if(i == 0)
                {
                    delta[i] = delta[i+1] * this.weights[i+1].Transpose * this.z[i];
                    djdw[i] = input.Transpose * delta[i];
                }
            }
            return djdw;
        }

        
        public void Train(double[] inputs, double[] target, int iterations = 50){
            for (int i = 0; i < iterations; i++)
            {
                for (int j = 0; j < inputs.Length; j++)
                {
                    ;
                }
            }
        }

        #endregion
    }
}