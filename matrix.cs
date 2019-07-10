using System;


namespace mnist{
    class matrix
    {
        #region variables
        /// <summary>
        /// Dimensions of the Matrix
        /// </summary>
        public int rows,columns;

        /// <summary>
        /// Storage array for the Matrix data.
        /// </summary>
        public double[,] data;

        Random rand = new Random();

        #endregion

        #region constructors
        /// <summary>
        /// Constructor to create a new Matrix while specifying the number of
        /// rows and columns.
        /// </summary>
        /// <param name="rows">The number of rows to initialise the Matrix with.</param>
        /// <param name="columns">The number of columns to initialise the Matrix with.</param>
        public matrix(int rows,int columns){
            this.rows = rows;
            this.columns = columns;
            data = new double[rows,columns];
        }

        /// <summary>
        /// Constructor to create a new Matrix with a given matrix;
        /// </summary>
        public matrix(double[,] m) : this(m.GetLength(0),m.GetLength(1))
        {
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    data[i,j] = m[i,j];
                }
            }
        }

        /// <summary>
        /// Constructor to create a new Matrix with a given vector;
        /// </summary>
        public matrix(double[] m) : this(m.GetLength(0), 1)
        {
            for (int j = 0; j < columns; j++)
            {
                data[j,0] = m[j];
            }
        }

        #endregion

        #region Properties
        /// <summary>
        /// Indicates whether or not this Matrix rows and columns dimensions are equal.
        /// </summary>
        public bool IsSquare => rows == columns;

        /// <summary>
        /// Get the transposed version of this Matrix (swap rows and columns)
        /// </summary>
        public matrix Transpose{
            get
            {
                matrix t = new matrix(columns,rows);
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < columns; j++)
                    {
                        t.data[j,i] = data[i,j];
                    }
                }
                return t;
            }
        }

        #endregion

        #region methods
        public void print_data(){
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    Console.Write(data[i,j] + " ");
                }
                Console.WriteLine();
            }
        }

        public void print_transpose(){
            for (int i = 0; i < columns; i++)
            {
                for (int j = 0; j < rows; j++)
                {
                    Console.Write(Transpose.data[i,j] + " ");
                }
                Console.WriteLine();
            }
        }

        /// <summary>
        /// Give a random value between -1 and 1 for each element in data.
        /// </summary>
        public void randomize_data(){
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    data[i,j] = rand.NextDouble() * 2 - 1;
                }
            }
        }

        /// <summary>
        /// Set each element in data to 1.
        /// </summary>
        public void weights_1(){
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    data[i,j] = 1;
                }
            }
        }

        /// <summary>
        /// Apply the sigmoid function over each element in data.
        /// </summary>
        public void ApplySigmoid(){
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    data[i,j] = 1/(1 + Math.Exp(-data[i,j]));
                }
            }
        }

        /// <summary>
        /// Apply the derivative sigmoid function over each element in data.
        /// </summary>
        public void ApplySigmoidPrime(){
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    data[i,j] = Math.Exp(-data[i,j])/Math.Pow((1 + Math.Exp(-data[i,j])),2);
                }
            }
        }

        /// <summary>
        /// Indicates if this Matrix has the same dimensions as another supplied Matrix.
        /// </summary>
        /// <param name="other">Another matrix object to compare this instance to.</param>
        /// <returns>true if both matrices have the same dimensions. Otherwise, false.</returns>
        public bool HaveSameDimensions(matrix other)
        {
            return (this.rows == other.rows) && (this.columns == other.columns);
        }

        public double sum(){
            double sum = 0;
            for (int i = 0; i < this.rows; i++)
            {
                for (int j = 0; j < this.columns; j++)
                {
                    sum += this.data[i,j];
                }
            }
            return sum;
        }

        #endregion

        #region operations
        /// <summary>
        /// Add two matrices together.
        /// </summary>
        /// <param name="m1">The first matrix object to add.</param>
        /// <param name="m2">The second matrix object to add.</param>
        /// <returns>The result of adding the two matrices together.</returns>
        /// <exception cref="Exception">Thrown when both matrices have
        /// different dimensions.</exception>
        public static matrix operator +(matrix m1, matrix m2){
            if (m1.HaveSameDimensions(m2))
            {
                matrix output = new matrix(m1.rows, m1.columns);
                for (int i = 0; i < m1.rows; i++)
                {
                    for (int j = 0; j < m1.columns; j++)
                    {
                        output.data[i,j] = m1.data[i,j] + m2.data[i,j];
                    }
                }
                return output;
            }
            else if (m1.rows == m2.rows && m2.columns == 1){
                matrix output = new matrix(m1.rows, m1.columns);
                for (int i = 0; i < m1.rows; i++)
                {
                    for (int j = 0; j < m1.columns; j++)
                    {
                        output.data[i,j] = m1.data[i,j] + m2.data[i,0];
                    }
                }
                return output;
            }
            else
            {
                throw new Exception("Cannot add two matrix objects whose dimensions do not match. (" + m1.rows + "," + m1.columns + ") and (" + m2.rows + "," + m2.columns + ")");
            }
        }
        
        /// <summary>
        /// Add a number to each element in a Matrix.
        /// </summary>
        /// <param name="number">The number to add to each element in a Matrix.</param>
        /// <param name="m">The matrix object to add numbers to.</param>
        /// <returns>The result of adding the number to each element in a Matrix.</returns>
        public static matrix operator +(double number, matrix m){
            matrix output = new matrix(m.rows, m.columns);
            for (int i = 0; i < m.rows; i++)
            {
                for (int j = 0; j < m.columns; j++)
                {
                    output.data[i,j] = m.data[i,j] + number;
                }
            }
            return output;
        }

        /// <summary>
        /// Add a number to each element in a Matrix.
        /// </summary>
        /// <param name="number">The number to add to each element in a Matrix.</param>
        /// <param name="m">The matrix object to add numbers to.</param>
        /// <returns>The result of adding the number to each element in a Matrix.</returns>
        public static matrix operator +(matrix m, double number){
            return number + m;
        }

        /// <summary>
        /// Multiply two matrices together.
        /// </summary>
        /// <param name="m1">An nxm dimension matrix object.</param>
        /// <param name="m2">An mxp dimension matrix object.</param>
        /// <returns>An nxp Matrix that is the product of m1 and m2.</returns>
        /// <exception cref="Exception">Thrown when the number of columns in the
        /// first Matrix don't match the number of rows in the second Matrix.</exception>
        public static matrix operator *(matrix m1, matrix m2){
            if (m1.columns == m2.rows)
            {
                matrix output = new matrix(m1.rows, m2.columns);
                double temp = 0;
                for (int i = 0; i < m1.rows; i++)
                {
                    for (int j = 0; j < m2.columns; j++)
                    {
                        temp = 0;
                        for (int k = 0; k < m1.columns; k++)
                        {
                            temp += m1.data[i,k] * m2.data[k,j];
                        }
                        output.data[i,j] = temp;
                    }
                }
                return output;
            }
            else
            {
                throw new Exception("Multiplication cannot be performed on matrices with these dimensions. (" + m1.rows + "," + m1.columns + ") and (" + m2.rows + "," + m2.columns + ") ");
            }
        }

        /// <summary>
        /// Multiply two matrices together.
        /// </summary>
        /// <param name="m1">An nxm dimension Matrix.</param>
        /// <param name="m2">An mxp dimension matrix object.</param>
        /// <returns>An nxp Matrix that is the product of m1 and m2.</returns>
        /// <exception cref="Exception">Thrown when the number of columns in the
        /// first Matrix don't match the number of rows in the second Matrix.</exception>
        public static matrix operator *(double[,] m1, matrix m2){
            if (m1.GetLength(1) == m2.rows)
            {
                matrix output = new matrix(m1.GetLength(0), m2.columns);
                double temp = 0;
                for (int i = 0; i < m1.GetLength(0); i++)
                {
                    for (int j = 0; j < m2.columns; j++)
                    {
                        temp = 0;
                        for (int k = 0; k < m1.GetLength(1); k++)
                        {
                            temp += m1[i,k] * m2.data[k,j];
                        }
                        output.data[i,j] = temp;
                    }
                }
                return output;
            }
            else
            {
                throw new Exception("Multiplication cannot be performed on matrices with these dimensions. (" + m1.GetLength(0) + "," + m1.GetLength(1) + ") and (" + m2.rows + "," + m2.columns + ")");
            }
        }

        /// <summary>
        /// Multiply two matrices together.
        /// </summary>
        /// <param name="m1">An nxm dimension matrix object.</param>
        /// <param name="m2">An mxp dimension Matrix.</param>
        /// <returns>An nxp Matrix that is the product of m1 and m2.</returns>
        /// <exception cref="Exception">Thrown when the number of columns in the
        /// first Matrix don't match the number of rows in the second Matrix.</exception>
        public static matrix operator *(matrix m1, double[,] m2){
            if (m1.columns == m2.GetLength(0))
            {
                matrix output = new matrix(m1.rows, m2.GetLength(1));
                double temp = 0;
                for (int i = 0; i < m1.rows; i++)
                {
                    for (int j = 0; j < m2.GetLength(1); j++)
                    {
                        temp = 0;
                        for (int k = 0; k < m1.columns; k++)
                        {
                            temp += m1.data[i,k] * m2[k,j];
                        }
                        output.data[i,j] = temp;
                    }
                }
                return output;
            }
            else
            {
                throw new Exception("Multiplication cannot be performed on matrices with these dimensions. (" + m1.rows + "," + m1.columns + ") and (" + m2.GetLength(0) + "," + m2.GetLength(1) + ")");
            }
        }

        /// <summary>
        /// Scalar multiplication of a Matrix.
        /// </summary>
        /// <param name="number">The number value to multiply each element of the Matrix by.</param>
        /// <param name="m">The matrix object to apply multiplication to.</param>
        /// <returns>A Matrix representing the scalar multiplication of scalar * m.</returns>
        public static matrix operator *(double number, matrix m){
            matrix output = new matrix(m.rows, m.columns);
            for (int i = 0; i < m.rows; i++)
            {
                for (int j = 0; j < m.columns; j++)
                {
                    output.data[i,j] = m.data[i,j] * number;
                }
            }
            return output;
        }
        
        /// <summary>
        /// Scalar multiplication of a Matrix.
        /// </summary>
        /// <param name="m">The matrix object to apply multiplication to.</param>
        /// <param name="number">The number value to multiply each element of the Matrix by.</param>
        /// <returns>A Matrix representing the scalar multiplication of m * scalar.</returns>
        public static matrix operator *(matrix m, double number)
        {
            // Same as above, but ensuring commutativity - i.e. (s * m) == (m * s).
            return number * m;
        }
        #endregion
    }
}

        /// <summary>
        /// 
        /// </summary>
        /// <param name=""> </param>
        /// <returns> </returns>