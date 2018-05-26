using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using DeepCat.Initialization;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

namespace DeepCatTests
{
    [TestClass]
    public class InitializationTests
    {
        [TestMethod]
        public void TestFixedInitialization()
        {
            var initialization = Initializations.Fixed();

            var initializedMatrix = initialization.Initialize(2, 2);

            var expectedMatrix = Matrix<double>.Build.DenseOfArray(new double[,] { { 0.11, 0.12 }, { 0.21, 0.22 } });

            Assert.AreEqual(initializedMatrix, expectedMatrix);
        }

        [TestMethod]
        public void TestZeroInitialization()
        {
            var initialization = Initializations.Zero();

            var initializedMatrix = initialization.Initialize(2, 2);

            var expectedMatrix = Matrix<double>.Build.Dense(2,2);

            Assert.AreEqual(initializedMatrix, expectedMatrix);
        }

        [TestMethod]
        public void TestRandomNormalInitialization()
        {
            
            var initialization = Initializations.RandomNormal();
            initialization.SetSeed(0);

            var initializedMatrix = initialization.Initialize(2, 2);

            var expectedMatrix = Matrix<double>.Build.Random(2, 2, new Normal(new Random(0)));

            Console.WriteLine(initializedMatrix);
            Console.WriteLine(expectedMatrix);

            Assert.AreEqual(initializedMatrix, expectedMatrix);
        }
    }
}
