using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace DeepCat.Activation
{
    public class ReluActivation : IActivation
    {
        public Matrix<double> Activate(Matrix<double> input)
        {
            var row = input.RowCount;
            var col = input.ColumnCount;

            var mat = input.ToColumnMajorArray();

            for(int i = 0; i < mat.Length; i++)
            {
                mat[i] = Math.Max(0, mat[i]);
            }

            return Matrix<double>.Build.DenseOfColumnMajor(row, col, mat);
        }

        public Matrix<double> ActivateDerivative(Matrix<double> input)
        {
            var row = input.RowCount;
            var col = input.ColumnCount;

            var mat = input.ToColumnMajorArray();

            for (int i = 0; i < mat.Length; i++)
            {
                mat[i] = mat[i] < 0 ? 0 : 1;
            }

            return Matrix<double>.Build.DenseOfColumnMajor(row, col, mat);
        }
    }
}
