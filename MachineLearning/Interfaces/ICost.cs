using System;

namespace MachineLearning
{
    /// <summary>
    /// Interfaces defines cost function.
    /// </summary>
    public interface ICost : ICloneable
    {
        /// <summary>
        /// Type of cost function.
        /// </summary>
        Utilities.CostType Type
        {
            get;
        }

        /// <summary>
        /// A total cumulative cost from last <see cref="ICost.Reset"/> method call.
        /// </summary>
        double TotalCost
        {
            get;
        }

        /// <summary>
        /// Evaluates cost value from given inputs. Increments <see cref="ICost.TotalCost"/> by the calculated cost amout.
        /// </summary>
        /// <param name="predicted">A predicted value to be compared.</param>
        /// <param name="correct">A correvt value to be compared.</param>
        /// <returns>A cost value from given inputs.</returns>
        double Function(double predicted, double correct);

        /// <summary>
        /// Performs derivative of the cost function from given inputs.
        /// </summary>
        /// <param name="predicted">A predicted value to be compared.</param>
        /// <param name="correct">A correvt value to be compared.</param>
        /// <returns>A defined derivate for the cost function.</returns>
        double Derivative(double predicted, double correct);

        /// <summary>
        /// Resets <see cref="ICost.TotalCost"/> to 0.
        /// </summary>
        void Reset();
    }
}
