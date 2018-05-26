using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepCat.Optimization
{
    public static class Optimizers
    {
        public static IOptimizer GradientDescent(double learningRate)
        {
            return new GradientDescentOptimizer(learningRate);
        }

    }
}
