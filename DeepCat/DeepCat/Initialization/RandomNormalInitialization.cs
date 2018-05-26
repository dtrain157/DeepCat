using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

namespace DeepCat.Initialization
{
    public class RandomNormalInitialization : IInitialization
    {
        private int? _seed;

        public void SetSeed(int seed)
        {
            _seed = seed;
        }

        public Matrix<double> Initialize(int row, int col)
        {
            Matrix<double> mat;
            if (_seed != null)
            {
                mat = Matrix<double>.Build.Random(row, col, new Normal(new Random((int)_seed)));
            }
            else
            {
                mat = Matrix<double>.Build.Random(row, col, new Normal());
            }

            return mat;
        }
    }
}
