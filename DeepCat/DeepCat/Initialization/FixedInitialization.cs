using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

namespace DeepCat.Initialization
{
    public class FixedInitialization : IInitialization
    {
        public Matrix<double> Initialize(int row, int col)
        {
            var mat = new double[row, col];

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    mat[i, j] = Math.Round((0.1 * (i + 1)) + (0.01 * (j + 1)), 3);
                }
            }

            
            return Matrix<double>.Build.DenseOfArray(mat);
        }

        public void SetSeed(int seed)
        {
            throw new NotImplementedException();
        }
    }
}
