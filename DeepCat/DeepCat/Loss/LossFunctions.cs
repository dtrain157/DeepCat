using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepCat.Loss
{
    public static class LossFunctions
    {
        public static ILoss CrossEntropy()
        {
            return new CrossEntropyLoss();
        }
    }
}
