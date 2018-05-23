using System;

namespace MachineLearning
{
    public interface IRegularization : ICloneable
    {
        Utilities.RegularizationType Type
        {
            get;
        }

        double TotalPenalty
        {
            get;
        }

        double Derivative(double input, bool forApplied = true);

        double Function(double input);

        void Reset();
    }
}
