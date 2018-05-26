using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepCat.Activation
{
    public static class Activations
    {
        public static IActivation Relu()
        {
            return new ReluActivation();
        }

        public static IActivation Sigmoid()
        {
            return new SigmoidActivation();
        }
    }
}
