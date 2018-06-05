using System;
using System.Collections.Generic;

namespace MachineLearning
{
    /// <summary>
    /// An interface representing Neural Network functionality.
    /// </summary>
    public interface INeuralLayer : ICloneable
    {
        /// <summary>
        /// A regularization class to be used in training.
        /// </summary>
        IRegularization Regularization
        {
            get; set;
        }

        /// <summary>
        /// Representds information about Neural Network as one-dimensional array-list. Can be used to store or load data.
        /// </summary>
        IList<double> FlatData
        {
            get; set;
        }

        /// <summary>
        /// Represents a dimensionality of the input matrix.
        /// </summary>
        int[] InputDimensions
        {
            get;
        }

        /// <summary>
        /// Represents a dimensionality of the output matrix.
        /// </summary>
        int[] OutputDimensions
        {
            get;
        }

        /// <summary>
        /// Performs various operations on input matrix to produce an output matrix.
        /// </summary>
        /// <param name="inputs">An input matrix. Must have the same dimensionality as <see cref="INeuralLayer.InputDimensions"/>.</param>
        /// <returns>An output matrix with the dimensionality of <see cref="INeuralLayer.OutputDimensions"/>.</returns>
        Matrix FeedForward(Matrix inputs);

        /// <summary>
        /// Performs a backpropagation, optimazing the internal operations to best fit previous input to output based on error.
        /// </summary>
        /// <param name="error">Represents error between correct output and predicted output.</param>
        /// <param name="learningRate">Represents a learning rate, defining how mush to correct internal operations.</param>
        /// <param name="regularizationRate">Represents a regularization rate, defining how mush to regularize internal operations.</param>
        /// <returns>A propagated error matrix to be used by previous <see cref="INeuralLayer"/> in <see cref="NeuralNetwork"/></returns>
        Matrix BackPropagation(Matrix error, double learningRate, double regularizationRate);
    }
}
