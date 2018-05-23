using System;

namespace MachineLearning
{
    public interface ICost : ICloneable
    {
        Utilities.CostType Type
        {
            get;
        }

        double TotalCost
        {
            get;
        }

        double Function(double predicted, double correct);

        double Derivative(double predicted, double correct);

        void Reset();
    }
}
