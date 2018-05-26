using DeepCat.Activation;
using DeepCat.Initialization;
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
    class Program
    {
        static void Main(string[] args)
        {
            
            var X = Matrix<double>.Build.Random(5, 100);
            var Y = Matrix<double>.Build.Random(1, 100);

            var test = Matrix<double>.Build.Random(5, 1);


            var model = new DeepCat();
            model.Add(new Dense(5, Activations.Relu(), weightInitializer: Initializations.RandomNormal()));
            model.Add(new Dense(5, Activations.Relu(), weightInitializer: Initializations.RandomNormal()));
            model.Add(new Dense(1, Activations.Sigmoid()));

            model.Compile(X.RowCount, LossFunctions.CrossEntropy(), Optimizers.GradientDescent(0.002));

            model.Fit(X, Y, 100);
            model.Predict(test);
            

            
            

            var x = 1;

        }
    }
}
