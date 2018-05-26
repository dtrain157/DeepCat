using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepCat.Initialization
{
    public static class Initializations
    {
        public static IInitialization Zero()
        {
            return new ZeroInitialization();
        }

        public static IInitialization RandomNormal()
        {
            return new RandomNormalInitialization();
        }

        public static IInitialization Fixed()
        {
            return new FixedInitialization();
        }
    }
}
