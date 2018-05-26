using DeepCat.Layers;
using DeepCat.Loss;
using DeepCat.Optimization;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepCat
{
    public class DeepCat
    {
        public List<ILayer> _layers = new List<ILayer>();
        public IOptimizer _optimizer;
        public ILoss _lossFunction;

        public void Add(ILayer layer)
        {
            _layers.Add(layer);
        }

        public void Compile(int inputSize, ILoss lossFunction, IOptimizer optimizer)
        {
            _optimizer = optimizer;
            _lossFunction = lossFunction;

            for (var i = 0; i < _layers.Count; i++)
            {
                int previousLayerSize;
                if (i == 0)
                {
                    previousLayerSize = inputSize;
                }
                else
                {
                    previousLayerSize = _layers[i - 1].LayerSize;
                }
                _layers[i].Compile(previousLayerSize);
                _layers[i].Optimizer = optimizer;
            }
        }

        public void Fit(Matrix<double> X, Matrix<double> Y, int epochs, int? batchSize = null)
        {
            var m = batchSize ?? X.ColumnCount;

            foreach (var layer in _layers)
            {
                layer.BatchSize = m;
            }


            for (int i = 0; i < epochs; i++)
            {
                Console.WriteLine(string.Format("Epoch: {0}", i + 1));
                for (var batch = 0; batch < X.ColumnCount / m; batch++)
                {
                    var xBatch = X.SubMatrix(0, X.RowCount, batch * m, m);
                    var yBatch = Y.SubMatrix(0, Y.RowCount, batch * m, m);
                    var yhat = ForwardPropegation(xBatch);
                    var batchCost = _lossFunction.CalculateCost(yBatch, yhat);
                    var dYhat = _lossFunction.CalculateCostDerivative(yBatch, yhat);
                    BackwardPropegation(xBatch, dYhat);
                    Update();
                    Console.WriteLine(string.Format("Batch: {0}; Cost: {1}", batch + 1, batchCost));
                }

                //handle the last batch, which might not be complete
                if (X.ColumnCount % m != 0)
                {
                    var xBatch = PadMatrix(X.SubMatrix(0, X.RowCount, (X.ColumnCount / m) * m, X.ColumnCount - (X.ColumnCount / m) * m), m);
                    var yBatch = PadMatrix(Y.SubMatrix(0, Y.RowCount, (X.ColumnCount / m) * m, Y.ColumnCount - (Y.ColumnCount / m) * m), m);
                    var yhat = ForwardPropegation(xBatch);
                    var batchCost = _lossFunction.CalculateCost(yBatch, yhat);
                    var dYhat = _lossFunction.CalculateCostDerivative(yBatch, yhat);
                    BackwardPropegation(xBatch, dYhat);
                    Update();
                }

            }

        }

        internal void Compile(object p1, object p2)
        {
            throw new NotImplementedException();
        }

        public Matrix<double> Predict(Matrix<double> X)
        {
            return ForwardPropegation(X);
        }

        private Matrix<double> ForwardPropegation(Matrix<double> X)
        {
            var A_prev = X;
            foreach (var layer in _layers)
            {
                A_prev = layer.Forward(A_prev);
            }

            return A_prev;
        }

        private void BackwardPropegation(Matrix<double> X, Matrix<double> dYhat)
        {
            var dA_next = dYhat;

            for (int i = _layers.Count - 1; i >= 0; i--)
            {
                Matrix<double> A_prev;
                if (i != 0)
                {
                    A_prev = _layers[i - 1].GetActivation();
                }
                else
                {
                    A_prev = X;
                }
                dA_next = _layers[i].Backward(A_prev, dA_next);
            }
        }

        private void Update()
        {
            foreach (var layer in _layers)
            {
                layer.Update();
            }
        }

        private Matrix<double> PadMatrix(Matrix<double> X, int m)
        {
            var padSize = X.ColumnCount - m;
            var zeroPadMatrix = Matrix<double>.Build.Dense(X.RowCount, padSize);
            return Matrix<double>.Build.DenseOfMatrixArray(new[,] { { X, zeroPadMatrix } });
        }

    }
}
