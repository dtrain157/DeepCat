using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepCat.Activation
{
    public interface IActivation
    {
        Matrix<double> Activate(Matrix<double> input);
        Matrix<double> ActivateDerivative(Matrix<double> input);
    }
}
