using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using DeepCat.Activation;
using DeepCat.Initialization;
using DeepCat.Optimization;

namespace DeepCat.Layers
{
    public class Dense : ILayer
    {
        private IActivation _activation;
        private IInitialization _weightInitializer;
        private IInitialization _biasInitializer;
        
        private bool _useBias;

        //ForwardProp 
        private Matrix<double> W;   //weights [n(l) x n(l-1)]
        private Matrix<double> A;   //activation [n(l) x m]
        private Matrix<double> B;   //bias [n(l) x m]
        private Matrix<double> Z;   //trigger [n(l) x m]

        private Matrix<double> initial_b;   //single bias [n(l) x 1]   -- used for initialization

        //BackProp
        private Matrix<double> dW;
        private Matrix<double> dA;   
        private Matrix<double> dB;   
        private Matrix<double> dZ;            

        public int LayerSize { get; private set; }
        public int BatchSize { get; set; }
        public IOptimizer Optimizer { get ; set; }
       

        public Dense(int units, IActivation activation, bool useBias = true, IInitialization weightInitializer = null, IInitialization biasInitializer = null)
        {
            LayerSize = units;

            _activation = activation;
            _weightInitializer = weightInitializer ?? Initializations.Zero();
            _biasInitializer = biasInitializer ?? Initializations.Zero();
            _useBias = useBias;
        }

        public void Compile(int previousLayerSize)
        {
            W = _weightInitializer.Initialize(LayerSize, previousLayerSize);
            initial_b = _biasInitializer.Initialize(LayerSize, 1);
        }

        public Matrix<double> Forward(Matrix<double> A_prev)
        {
            if (B == null)
            {
                B = BuildBiasMatrix(initial_b);
            }

            Z = W * A_prev + B;
            A = _activation.Activate(Z);

            return A;
        }

        public Matrix<double> Backward(Matrix<double> A_prev, Matrix<double> dA_next)
        {
            dZ = ElementWiseMultiply(dA_next, _activation.ActivateDerivative(Z));
            dW = (1 / (double)BatchSize) * (dZ * A_prev.Transpose());
            dB = (1 / (double)BatchSize) * BuildBiasMatrix(Matrix<double>.Build.DenseOfColumnVectors(dZ.RowSums()));
            dA = W.Transpose() * dZ;

            return dA;
        }

        private Matrix<double> BuildBiasMatrix(Matrix<double> initial_b)
        {
            var B = initial_b;
            for(var i = 1; i < BatchSize; i++)
            {
               B = Matrix<double>.Build.DenseOfMatrixArray(matrices: new Matrix<double>[,] { { B, initial_b } });
            }

            return B;
        }

        private Matrix<double> ElementWiseMultiply(Matrix<double> X, Matrix<double> Y)
        {
            var xCols = X.AsColumnMajorArray();
            var yCols = Y.AsColumnMajorArray();
            var sum = new double[xCols.Length];

            for (int i = 0; i < xCols.Length; i++)
            {
                sum[i] = xCols[i] * yCols[i];
            }

            return Matrix<double>.Build.DenseOfColumnMajor(X.RowCount, X.ColumnCount, sum);
        }

        public void Update()
        {
            W = Optimizer.OptimizeWeights(W, dW);
            B = Optimizer.OptimizeBias(B, dB);
        }

        public Matrix<double> GetActivation()
        {
            return A;
        }
    }


}
