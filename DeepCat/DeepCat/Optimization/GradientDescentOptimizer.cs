using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace DeepCat.Optimization
{
    public class GradientDescentOptimizer : IOptimizer
    {
        private double _learningRate;

        public GradientDescentOptimizer(double learningRate)
        {
            _learningRate = learningRate;
        }

        public Matrix<double> OptimizeBias(Matrix<double> B, Matrix<double> dB)
        {
            return B - _learningRate * dB;
        }

        public Matrix<double> OptimizeWeights(Matrix<double> W, Matrix<double> dW)
        {
            return W - _learningRate * dW;
        }
    }
}
