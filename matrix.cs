using System;


namespace generic{
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
        public double[][] data;

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
            data = new double[rows][];
            for (int i = 0; i < rows; i++)
            {
                data[i] = new double[columns];
            }
        }

        /// <summary>
        /// Constructor to create a new Matrix with a given matrix;
        /// </summary>
        public matrix(double[][] m) : this(m.Length,m[0].Length)
        {
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    data[i][j] = m[i][j];
                }
            }
        }

        /// <summary>
        /// Constructor to create a new Matrix with a given vector;
        /// </summary>
        public matrix(double[] m) : this(1,m.Length)
        {
            for (int j = 0; j < rows; j++)
            {
                data[0][j] = m[j];
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
                        t.data[j][i] = data[i][j];
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
                    Console.Write(data[i][j] + " ");
                }
                Console.WriteLine();
            }
            Console.WriteLine();
        }

        public void print_transpose(){
            for (int i = 0; i < columns; i++)
            {
                for (int j = 0; j < rows; j++)
                {
                    Console.Write(Transpose.data[i][j] + " ");
                }
                Console.WriteLine();
            }
            Console.WriteLine();
        }

        /// <summary>
        /// Give a random value between -1 and 1 for each element in data.
        /// </summary>
        public void randomize_data(){
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    data[i][j] = rand.NextDouble() * 2 - 1;
                }
            }
        }

        public void size(){
            Console.WriteLine(rows + " , " + columns);
        }

        public static void size(double[][] x){
            Console.WriteLine(x.Length + " , " + x[0].Length);
        }

        public static void size(double[] x){
            Console.WriteLine(x.Length);
        }

        /// <summary>
        /// Set each element in data to 1.
        /// </summary>
        public void fill_data(double number){
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    data[i][j] = number;
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
                    data[i][j] = 1/(1 + Math.Exp(-data[i][j]));
                }
            }
        }

        /// <summary>
        /// Apply the sigmoid function over z.
        /// </summary>
        public static double Sigmoid(double z){
            return 1/(1+Math.Exp(-z));
        }

        public static matrix Sigmoid(matrix z){
            matrix result = new matrix(z.rows,z.columns);
            for (int i = 0; i < z.rows; i++)
            {
                for (int j = 0; j < z.columns; j++)
                {
                    result.data[i][j] = 1/(1 + Math.Exp(-z.data[i][j]));
                }
            }
            return result;
        }

        /// <summary>
        /// Apply the derivative sigmoid function over each element in data.
        /// </summary>
        public void ApplySigmoidDerivate(){
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    data[i][j] = Sigmoid(data[i][j]) * (1 - Sigmoid(data[i][j])); 
                }
            }
        }
        public static double SigmoidDerivate(double z){
            return Sigmoid(z) * (1 - Sigmoid(z));
        }
        
        public static matrix SigmoidDerivate(matrix z){
            matrix result = new matrix(z.rows,z.columns);
            for (int i = 0; i < z.rows; i++)
            {
                for (int j = 0; j < z.columns; j++)
                {
                    result.data[i][j] = Sigmoid(z.data[i][j]) * (1 - Sigmoid(z.data[i][j]));
                }
            }
            return result;
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
                    sum += this.data[i][j];
                }
            }
            return sum;
        }

        /// <summary>
        /// Sums all alements in matrix.
        /// </summary>
        public static double sum(matrix x){
            double sum = 0;
            for (int i = 0; i < x.rows; i++)
            {
                for (int j = 0; j < x.columns; j++)
                {
                    sum += x.data[i][j];
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
                        output.data[i][j] = m1.data[i][j] + m2.data[i][j];
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
                        output.data[i][j] = m1.data[i][j] + m2.data[i][0];
                    }
                }
                return output;
            }
            else if (m1.columns == m2.columns && m2.rows == 1){
                matrix output = new matrix(m1.rows, m1.columns);
                for (int i = 0; i < m1.rows; i++)
                {
                    for (int j = 0; j < m1.columns; j++)
                    {
                        output.data[i][j] = m1.data[i][j] + m2.data[0][j];
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
                    output.data[i][j] = m.data[i][j] + number;
                }
            }
            return output;
        }

        public static matrix operator +(matrix m, double number){
            return number + m;
        }

        /// <summary>
        /// Subtract two matrices together.
        /// </summary>
        /// <param name="m1">The first matrix object to subtract.</param>
        /// <param name="m2">The second matrix object to subtract.</param>
        /// <returns>The result of subtracting the two matrices together.</returns>
        /// <exception cref="Exception">Thrown when both matrices have
        /// different dimensions.</exception>
        public static matrix operator -(matrix m1, matrix m2){
            if (m1.HaveSameDimensions(m2))
            {
                matrix output = new matrix(m1.rows, m1.columns);
                for (int i = 0; i < m1.rows; i++)
                {
                    for (int j = 0; j < m1.columns; j++)
                    {
                        output.data[i][j] = m1.data[i][j] - m2.data[i][j];
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
                        output.data[i][j] = m1.data[i][j] - m2.data[i][0];
                    }
                }
                return output;
            }
            else if (m1.columns == m2.columns && m2.rows == 1){
                matrix output = new matrix(m1.rows, m1.columns);
                for (int i = 0; i < m1.rows; i++)
                {
                    for (int j = 0; j < m1.columns; j++)
                    {
                        output.data[i][j] = m1.data[i][j] - m2.data[0][j];
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
        /// Subtract a number to each element in a Matrix.
        /// </summary>
        /// <param name="number">The number to subtract to each element in a Matrix.</param>
        /// <param name="m">The matrix object to subtract numbers to.</param>
        /// <returns>The result of subtracting the number to each element in a Matrix.</returns>
        public static matrix operator -(double number, matrix m){
            matrix output = new matrix(m.rows, m.columns);
            for (int i = 0; i < m.rows; i++)
            {
                for (int j = 0; j < m.columns; j++)
                {
                    output.data[i][j] = number - m.data[i][j];
                }
            }
            return output;
        }

        public static matrix operator -(matrix m, double number){
            matrix output = new matrix(m.rows, m.columns);
            for (int i = 0; i < m.rows; i++)
            {
                for (int j = 0; j < m.columns; j++)
                {
                    output.data[i][j] = m.data[i][j] - number;
                }
            }
            return output;
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
                            temp += m1.data[i][k] * m2.data[k][j];
                        }
                        output.data[i][j] = temp;
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
        public static matrix operator *(double[][] m1, matrix m2){
            if (m1[0].Length == m2.rows)
            {
                matrix output = new matrix(m1.Length, m2.columns);
                double temp = 0;
                for (int i = 0; i < m1.Length; i++)
                {
                    for (int j = 0; j < m2.columns; j++)
                    {
                        temp = 0;
                        for (int k = 0; k < m1[0].Length; k++)
                        {
                            temp += m1[i][k] * m2.data[k][j];
                        }
                        output.data[i][j] = temp;
                    }
                }
                return output;
            }
            else
            {
                throw new Exception("Multiplication cannot be performed on matrices with these dimensions. (" + m1.Length + "," + m1[0].Length + ") and (" + m2.rows + "," + m2.columns + ")");
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
        public static matrix operator *(matrix m1, double[][] m2){
            if (m1.columns == m2.Length)
            {
                matrix output = new matrix(m1.rows, m2[0].Length);
                double temp = 0;
                for (int i = 0; i < m1.rows; i++)
                {
                    for (int j = 0; j < m2[0].Length; j++)
                    {
                        temp = 0;
                        for (int k = 0; k < m1.columns; k++)
                        {
                            temp += m1.data[i][k] * m2[k][j];
                        }
                        output.data[i][j] = temp;
                    }
                }
                return output;
            }
            else
            {
                throw new Exception("Multiplication cannot be performed on matrices with these dimensions. (" + m1.rows + "," + m1.columns + ") and (" + m2.Length + "," + m2[0].Length + ")");
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
                    output.data[i][j] = m.data[i][j] * number;
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

        /// <summary>
        /// Multiply element wise two matrices together.
        /// </summary>
        /// <param name="m1">An nxm dimension matrix object.</param>
        /// <param name="m2">An mxp dimension matrix object.</param>
        /// <returns>An nxm Matrix that is the product of m1 and m2.</returns>
        /// <exception cref="Exception">Thrown when the number of columns and rows in the
        /// first Matrix don't match the number of columns and rows in the second Matrix.</exception>
        public static matrix elementWiseMultiplication(matrix m1, matrix m2){
            if (m1.HaveSameDimensions(m2))
            {
                matrix output = new matrix(m1.rows, m1.columns);
                for (int i = 0; i < m1.rows; i++)
                {
                    for (int j = 0; j < m1.columns; j++)
                    {
                        output.data[i][j] = m1.data[i][j] * m2.data[i][j];
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
                        output.data[i][j] = m1.data[i][j] * m2.data[i][0];
                    }
                }
                return output;
            }
            else if (m1.columns == m2.columns && m2.rows == 1){
                matrix output = new matrix(m1.rows, m1.columns);
                for (int i = 0; i < m1.rows; i++)
                {
                    for (int j = 0; j < m1.columns; j++)
                    {
                        output.data[i][j] = m1.data[i][j] * m2.data[0][j];
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
        /// Perform an element wise raise to the given matrix.
        /// </summary>
        /// <param name="m1">An nxm dimension matrix object. </param>
        /// <param name="nr">The power .</param>
        /// <returns>An nxm Matrix that is the result of raising m1 to the power nr. </returns>
        public static matrix operator^(matrix m1, double nr){
            double temp = 0;
            for (int i = 0; i < m1.rows; i++)
            {
                for (int j = 0; j < m1.columns; j++)
                {
                    temp = m1.data[i][j];
                    for (int k = 1; k < nr; k++)
                    {
                        m1.data[i][j] = m1.data[i][j] * temp;
                    }
                }
            }
            return m1;
        }
        #endregion
        
    }
}

        /// <summary>
        /// 
        /// </summary>
        /// <param name=""> </param>
        /// <returns> </returns>
        