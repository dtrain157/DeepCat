using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepCat.Initialization
{
    public interface IInitialization
    {
        Matrix<double> Initialize(int row, int col);
        void SetSeed(int seed);
    }
}
