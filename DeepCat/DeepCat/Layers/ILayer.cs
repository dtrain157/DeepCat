using DeepCat.Optimization;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepCat.Layers
{
    public interface ILayer
    {
        int LayerSize { get; }
        int BatchSize { get; set; }
        IOptimizer Optimizer { get; set; }
        void Compile(int previousLayerSize);
        Matrix<double> Forward(Matrix<double> A_prev);
        Matrix<double> Backward(Matrix<double> A_prev, Matrix<double> dA_next);
        Matrix<double> GetActivation();
        void Update();
    }
}
