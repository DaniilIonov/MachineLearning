using System;

namespace MachineLearning
{

    /// <summary>
    /// This class is used to add Map and Absolute methods to any double objects.
    /// </summary>
    public static class ExtensionMethods
    {
        /// <summary>
        /// Re-maps a number from one range to another.
        /// </summary>
        /// <param name="value">Value to map.</param>
        /// <param name="fromSource">The lower bound of the value’s current range</param>
        /// <param name="toSource">The upper bound of the value’s current range</param>
        /// <param name="fromTarget">The lower bound of the value’s target range</param>
        /// <param name="toTarget">The upper bound of the value’s target range</param>
        /// <returns>Mapped value.</returns>
        public static double Map(this double value, double fromSource, double toSource, double fromTarget, double toTarget)
        {
            return (value - fromSource) / (toSource - fromSource) * (toTarget - fromTarget) + fromTarget;
        }

        /// <summary>
        /// Returns the absolute value of a double-precision floating-point number.
        /// </summary>
        /// <param name="value">A number that is greater than or equal to System.Double.MinValue, but less than or equal to System.Double.MaxValue.</param>
        /// <returns>A double-precision floating-point number, x, such that 0 ≤ x ≤System.Double.MaxValue.</returns>
        public static double Absolute(this double value)
        {
            return Math.Abs(value);
        }
    }
}
