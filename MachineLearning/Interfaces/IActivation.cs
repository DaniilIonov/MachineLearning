using System;

namespace MachineLearning
{
    public interface IActivation : ICloneable
    {
        Utilities.ActivationFunction Type
        {
            get;
        }

        /// <summary>
        /// Represents operation function to be performed.
        /// </summary>
        /// <param name="input">An input value for operation.</param>
        /// <returns>Modified input value.</returns>
        double Function(double input);

        /// <summary>
        /// Represents operation function derivative at the particular value.
        /// </summary>
        /// <param name="input">An input value to find a defined derivative.</param>
        /// <returns>A defined derivative of the function at the particular value.</returns>
        double Derivative(double input);
    }
}
