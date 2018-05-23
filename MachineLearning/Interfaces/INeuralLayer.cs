using System;
using System.Collections.Generic;

namespace MachineLearning
{
    /// <summary>
    /// Defines a Neural Layer Interface to be used in <see cref="NeuralNetworkOld"/> class.
    /// </summary>
    public interface INeuralLayer : ICloneable
    {
        IRegularization Regularization
        {
            get; set;
        }

        IActivation Activation
        {
            get; set;
        }

        IList<double> FlatData
        {
            get; set;
        }

        int[] InputDimensions
        {
            get;
        }

        int[] OutputDimensions
        {
            get;
        }

        Matrix FeedForward(Matrix inputs);

        Matrix BackPropagation(Matrix error);

        void CorrectWeights(double learningRate, double regularizationRate);
    }
}
