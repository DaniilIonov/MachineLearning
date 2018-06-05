using System;

namespace MachineLearning
{
    /// <summary>
    /// Interface that defines a regularization.
    /// </summary>
    public interface IRegularization : ICloneable
    {
        /// <summary>
        /// A type of regularization.
        /// </summary>
        Utilities.RegularizationType Type
        {
            get;
        }

        /// <summary>
        /// Represents total cumulative regularization from last <see cref="IRegularization.Reset"/> method call.
        /// </summary>
        double TotalPenalty
        {
            get;
        }

        /// <summary>
        /// An regularization function to be performed on the input. Increments <see cref="IRegularization.TotalPenalty"/> by the regularized amout.
        /// </summary>
        /// <param name="input">A value to be regularized.</param>
        /// <returns>A regularized value.</returns>
        double Function(double input);

        /// <summary>
        /// A defined derivative of regularization function at the given input value.
        /// </summary>
        /// <param name="input">An input to be used in derivative calculations.</param>
        /// <param name="forApplied">A boolean value representing if the input value is already regularized or not. Default is true, meaning that the input is already regularized.
        /// </param>
        /// <returns></returns>
        double Derivative(double input, bool forApplied = true);

        /// <summary>
        /// Resets <see cref="IRegularization.TotalPenalty"/> to 0.
        /// </summary>
        void Reset();
    }
}
