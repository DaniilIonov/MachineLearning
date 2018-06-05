using System;

namespace MachineLearning
{
    /// <summary>
    /// Interface that defines activation function.
    /// </summary>
    public interface IActivation : ICloneable
    {
        /// <summary>
        /// Type of Activation Function.
        /// </summary>
        Utilities.ActivationFunction Type
        {
            get;
        }

        /// <summary>
        /// Represents activation function to be performed.
        /// </summary>
        /// <param name="input">An input value for activation.</param>
        /// <returns>Modified input value.</returns>
        double Function(double input);

        /// <summary>
        /// Represents activation function derivative at the particular value.
        /// </summary>
        /// <param name="input">An input value to find a defined derivative.</param>
        /// <returns>A defined derivative of the function at the particular value.</returns>
        double Derivative(double input);
    }
}
