using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepCat.Loss
{
    public interface ILoss
    {
        double CalculateCost(Matrix<double> yBatch, Matrix<double> yhat);
        Matrix<double> CalculateCostDerivative(Matrix<double> yBatch, Matrix<double> yhat);
    }
}
