using System;

namespace MachineLearning
{
    public class NoRegularization : IRegularization
    {
        private Utilities.RegularizationType _type = Utilities.RegularizationType.None;
        private double _totalPenalty;

        public Utilities.RegularizationType Type
        {
            get
            {
                return this._type;
            }
        }

        public double TotalPenalty
        {
            get
            {
                return this._totalPenalty;
            }
            set
            {
                this._totalPenalty = value;
            }
        }

        public object Clone()
        {
            return new NoRegularization()
            {
                TotalPenalty = this.TotalPenalty
            };
        }

        public double Derivative(double input, bool forApplied = true)
        {
            return 0.0;
        }

        public double Function(double input)
        {
            return 0.0;
        }

        public void Reset()
        {
            this.TotalPenalty = 0.0;
        }
    }

    public class L1Regularization : IRegularization
    {
        private Utilities.RegularizationType _type = Utilities.RegularizationType.L1;
        private double _totalPenalty;

        public Utilities.RegularizationType Type
        {
            get
            {
                return this._type;
            }
        }

        public double TotalPenalty
        {
            get
            {
                return this._totalPenalty;
            }
            private set
            {
                this._totalPenalty = value;
            }
        }

        public object Clone()
        {
            return new L1Regularization()
            {
                TotalPenalty = this.TotalPenalty
            };
        }

        public double Derivative(double input, bool forApplied = true)
        {
            if (input < 0.0)
            {
                return -1.0;
            }
            else if (input > 0.0)
            {
                return 1.0;
            }
            else
            {
                return 0.0;
            }
        }

        public double Function(double input)
        {
            double val = Math.Abs(input);
            this.TotalPenalty += val;
            return val;
        }

        public void Reset()
        {
            this.TotalPenalty = 0.0;
        }
    }

    public class L2Regularization : IRegularization
    {
        private Utilities.RegularizationType _type = Utilities.RegularizationType.L2;
        private double _totalPenalty;

        public Utilities.RegularizationType Type
        {
            get
            {
                return this._type;
            }
        }

        public double TotalPenalty
        {
            get
            {
                return this._totalPenalty;
            }
            private set
            {
                this._totalPenalty = value;
            }
        }

        public object Clone()
        {
            return new L2Regularization()
            {
                TotalPenalty = this.TotalPenalty
            };
        }

        public double Derivative(double input, bool forApplied = true)
        {
            return input;
        }

        public double Function(double input)
        {
            double val = input * input / 2.0;
            this.TotalPenalty += val;
            return val;
        }

        public void Reset()
        {
            this.TotalPenalty = 0.0;
        }
    }

    public class QuadraticCost : ICost
    {
        private Utilities.CostType _type = Utilities.CostType.Quadratic;
        private double _total;

        public Utilities.CostType Type
        {
            get
            {
                return this._type;
            }
        }

        public double TotalCost
        {
            get
            {
                return this._total;
            }
            private set
            {
                this._total = value;
            }
        }

        public object Clone()
        {
            return new QuadraticCost()
            {
                TotalCost = this.TotalCost
            };
        }

        public double Derivative(double predicted, double correct)
        {
            return (predicted - correct);
        }

        public double Function(double predicted, double correct)
        {
            double val = (predicted - correct) * (predicted - correct) * 0.5;
            this.TotalCost += val;
            return val;
        }

        public void Reset()
        {
            this.TotalCost = 0.0;
        }
    }

    public class CrossEntropyCost : ICost
    {
        private Utilities.CostType _type = Utilities.CostType.Quadratic;
        private double _total;

        public Utilities.CostType Type
        {
            get
            {
                return this._type;
            }
        }

        public double TotalCost
        {
            get
            {
                return this._total;
            }
            private set
            {
                this._total = value;
            }
        }

        public object Clone()
        {
            return new CrossEntropyCost()
            {
                TotalCost = this.TotalCost
            };
        }

        public double Derivative(double predicted, double correct)
        {
            return (predicted - correct) / (predicted - predicted * predicted);
        }

        public double Function(double predicted, double correct)
        {
            double val = ((correct - 1.0) * Math.Log(1.0 - predicted) - correct * Math.Log(predicted));
            this.TotalCost += val;
            return val;
        }

        public void Reset()
        {
            this.TotalCost = 0.0;
        }
    }

    public class Identity : IActivation
    {
        private Utilities.ActivationFunction _type = Utilities.ActivationFunction.Identity;

        public Utilities.ActivationFunction Type
        {
            get
            {
                return this._type;
            }
        }

        public double Function(double input)
        {
            return input;
        }

        public double Derivative(double input)
        {
            return 1.0;
        }

        public object Clone()
        {
            return new Identity();
        }
    }

    public class Logistic : IActivation
    {
        private Utilities.ActivationFunction _type = Utilities.ActivationFunction.Logistic;

        public Utilities.ActivationFunction Type
        {
            get
            {
                return this._type;
            }
        }

        public double Function(double input)
        {
            return 1 / (1 + Math.Exp(-1.0 * input));
        }

        public double Derivative(double input)
        {
            return (input * (1.0 - input));
        }

        public object Clone()
        {
            return new Logistic();
        }
    }

    public class TanH : IActivation
    {
        private Utilities.ActivationFunction _type = Utilities.ActivationFunction.TanH;

        public Utilities.ActivationFunction Type
        {
            get
            {
                return this._type;
            }
        }

        public double Function(double input)
        {
            return Math.Tanh(input);
        }

        public double Derivative(double input)
        {
            return (1.0 - input * input);
        }

        public object Clone()
        {
            return new TanH();
        }
    }

    public class ReLU : IActivation
    {
        private Utilities.ActivationFunction _type = Utilities.ActivationFunction.ReLU;

        public Utilities.ActivationFunction Type
        {
            get
            {
                return this._type;
            }
        }

        public object Clone()
        {
            return new ReLU();
        }

        public double Derivative(double input)
        {
            if (input < 0.0)
            {
                return 0.0;
            }
            else
            {
                return 1.0;
            }
        }

        public double Function(double input)
        {
            if (input < 0.0)
            {
                return 0.0;
            }
            else
            {
                return input;
            }
        }
    }

    public class LeakyReLU : IActivation
    {
        private Utilities.ActivationFunction _type = Utilities.ActivationFunction.LeakyReLU;

        public Utilities.ActivationFunction Type
        {
            get
            {
                return this._type;
            }
        }

        public object Clone()
        {
            return new LeakyReLU();
        }

        public double Derivative(double input)
        {
            if (input < 0.0)
            {
                return 0.01;
            }
            else
            {
                return 1.0;
            }
        }

        public double Function(double input)
        {
            if (input < 0.0)
            {
                return 0.01 * input;
            }
            else
            {
                return input;
            }
        }
    }
}
