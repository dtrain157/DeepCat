using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepCat.Optimization
{
    public interface IOptimizer
    {

        Matrix<double> OptimizeWeights(Matrix<double> W, Matrix<double> dW);
        Matrix<double> OptimizeBias(Matrix<double> B, Matrix<double> dB);

    }
}
