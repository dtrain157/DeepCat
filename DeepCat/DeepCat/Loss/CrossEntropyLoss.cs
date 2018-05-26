using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace DeepCat.Loss
{
    public class CrossEntropyLoss : ILoss
    {
        public double CalculateCost(Matrix<double> yBatch, Matrix<double> yhat)
        {
            double m = yBatch.ColumnCount;
            var yBatchFlat = yBatch.AsColumnMajorArray();
            var yHatFlat = yhat.AsColumnMajorArray();

            var loss = 0.0;

            for (int i = 0; i < yBatchFlat.Length; i++)
            {
                var loss_i = yBatchFlat[i] * Math.Log(yHatFlat[i]) + (1 - yBatchFlat[i]) * Math.Log(1 - yHatFlat[i]);
                loss = loss + loss_i;
            }

            return (-1) * (1 / m) * loss; 
        }

        public Matrix<double> CalculateCostDerivative(Matrix<double> yBatch, Matrix<double> yhat)
        {
            var row = yBatch.RowCount;
            var col = yBatch.ColumnCount;
            var yBatchFlat = yBatch.AsColumnMajorArray();
            var yHatFlat = yhat.AsColumnMajorArray();

            var matflat = new double[yBatchFlat.Length];

            for (int i = 0; i < yBatchFlat.Length; i++)
            {
                matflat[i] = (-1) * (yBatchFlat[i] / yHatFlat[i]) - ((1 - yBatchFlat[i]) / (1 - yHatFlat[i]));
            }

            return Matrix<double>.Build.DenseOfColumnMajor(row, col, matflat);
        }
    }
}
