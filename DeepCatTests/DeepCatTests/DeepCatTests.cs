using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using DeepCat.Initialization;
using MathNet.Numerics.LinearAlgebra;
using DeepCat.Layers;
using DeepCat.Activation;
using DeepCat.Loss;
using DeepCat.Optimization;

namespace DeepCatTests
{
    [TestClass]
    public class DeepCatTests
    {
        [TestMethod]
        public void LogisticRegression()
        {
            var X = Matrix<double>.Build.DenseOfArray(new double[,] { { 1 }, { 2 } });
            var Y = Matrix<double>.Build.DenseOfArray(new double[,] { { 1 } });

            var model = new DeepCat.DeepCat();
            model.Add(new Dense(1, Activations.Sigmoid(), weightInitializer: Initializations.Fixed()));
            model.Compile(X.RowCount, LossFunctions.CrossEntropy(), Optimizers.GradientDescent(0.02));
            model.Fit(X, Y, 1);

            var a = model.Predict(X);
            a[0, 0] = Math.Round(a[0, 0], 8);

            var expectedResult = Matrix<double>.Build.DenseOfArray(new double[,] { { 0.59859297 } });

            Assert.AreEqual(a, expectedResult);
        }

       
    }
}
