using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace DeepCat.Initialization
{
    public class ZeroInitialization : IInitialization
    {
        public Matrix<double> Initialize(int row, int col)
        {
            return Matrix<double>.Build.Dense(row, col);
        }

        public void SetSeed(int seed)
        {
            throw new NotImplementedException();
        }
    }
}
