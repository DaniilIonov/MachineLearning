using System;
using System.Collections.Generic;
using System.Linq;

namespace MachineLearning
{
    public class FullyConnectedLayer : INeuralLayer
    {
        private int[] _inputDimensions;
        private int[] _outputDimensions;
        private Matrix _weights;
        private Matrix _biases;
        private IActivation _activation = new Identity();
        private IRegularization _regularization = new NoRegularization();
        private Matrix _inputs;
        private Matrix _outputs;
        private Matrix _gradients;
        private Matrix _deltaWeights;
        private Matrix _error;

        public int[] InputDimensions
        {
            get
            {
                return this._inputDimensions;
            }
            private set
            {
                this._inputDimensions = value.ToArray();
            }
        }

        public int[] OutputDimensions
        {
            get
            {
                return this._outputDimensions;
            }
            private set
            {
                this._outputDimensions = value.ToArray();
            }
        }

        public Matrix Weights
        {
            get
            {
                return this._weights;
            }
            private set
            {
                this._weights = value;
            }
        }

        public Matrix Biases
        {
            get
            {
                return this._biases;
            }
            private set
            {
                this._biases = value;
            }
        }

        public IActivation Activation
        {
            get
            {
                return this._activation;
            }
            set
            {
                this._activation = value;
            }
        }

        public IRegularization Regularization
        {
            get
            {
                return this._regularization;
            }
            set
            {
                this._regularization = value;
            }
        }

        public Matrix Inputs
        {
            get
            {
                return this._inputs;
            }
            private set
            {
                this._inputs = value;
            }
        }

        public Matrix Outputs
        {
            get
            {
                return this._outputs;
            }
            private set
            {
                this._outputs = value;
            }
        }

        public Matrix Error
        {
            get
            {
                return this._error;
            }
            private set
            {
                this._error = value;
            }
        }

        public Matrix Gradients
        {
            get
            {
                return this._gradients;
            }
            private set
            {
                this._gradients = value;
            }
        }

        public Matrix DeltaWeights
        {
            get
            {
                return this._deltaWeights;
            }
            private set
            {
                this._deltaWeights = value;
            }
        }

        public IList<double> FlatData
        {
            get
            {
                List<double> data = new List<double>();

                data.AddRange(this.Weights.GetFlatData().Concat(this.Biases.GetFlatData()));

                return data;
            }
            set
            {
                int index = 0;

                this.Weights.Map(() => value[index++]);
                this.Biases.Map(() => value[index++]);
            }
        }

        public FullyConnectedLayer(int[] inputDimensions, int[] outputDimensions)
        {
            this.InputDimensions = inputDimensions.ToArray();
            this.OutputDimensions = outputDimensions.ToArray();

            this.Biases = new Matrix(this.OutputDimensions);
            int[] weightsDimensions = Matrix.GenerateInfoForMatrixMultiplication(this.OutputDimensions, this.InputDimensions).Item1;
            Randomize();
        }

        public FullyConnectedLayer(int inputs, int outputs)
        {
            this.InputDimensions = new int[] { inputs };
            this.OutputDimensions = new int[] { outputs };

            this.Biases = new Matrix(this.OutputDimensions);
            int[] weightsDimensions = Matrix.GenerateInfoForMatrixMultiplication(this.OutputDimensions, this.InputDimensions).Item1;
            this.Weights = new Matrix(weightsDimensions);
            Randomize();
        }

        public FullyConnectedLayer Randomize()
        {
            this.Biases.Map(() => ThreadSafeRandom.Gaussian());
            this.Weights.Map(() => ThreadSafeRandom.Gaussian(0.0, 1.0 / Math.Sqrt(new Matrix(this.InputDimensions).Data.Length)));

            return this;
        }

        public Matrix FeedForward(Matrix inputs)
        {
            this.Inputs = inputs.Clone() as Matrix;
            this.Outputs = this.Weights * this.Inputs;
            this.Outputs += this.Biases;
            this.Outputs.Map((double value) => this.Activation.Function(value));
            return this.Outputs.Clone() as Matrix;
        }

        public Matrix BackPropagation(Matrix error)
        {
            this.Error = error.Clone() as Matrix;
            this.Gradients = (this.Outputs.Clone() as Matrix).Map((double value) => this.Activation.Function(value));
            this.Gradients ^= this.Error;

            this.DeltaWeights = this.Gradients * !this.Inputs;

            return !this.Weights * this.Error;
        }

        public void CorrectWeights(double learningRate, double regularizationRate)
        {
            this.Biases -= (this.Gradients * learningRate);

            Matrix regularization = (this.Weights.Clone() as Matrix).Map((double weight) =>
            {
                this.Regularization.Function(weight);
                return this.Regularization.Derivative(weight);
            });

            this.Weights -= (this.DeltaWeights + (regularization * regularizationRate)) * learningRate;
        }

        public object Clone()
        {
            return new FullyConnectedLayer(this.InputDimensions, this.OutputDimensions)
            {
                Weights = this.Weights.Clone() as Matrix,
                Biases = this.Biases.Clone() as Matrix,
                Activation = this.Activation.Clone() as IActivation,
                Regularization = this.Regularization.Clone() as IRegularization,
                Inputs = this.Inputs.Clone() as Matrix,
                Outputs = this.Outputs.Clone() as Matrix,
                Gradients = this.Gradients.Clone() as Matrix,
                DeltaWeights = this.DeltaWeights.Clone() as Matrix,
                Error = this.Error.Clone() as Matrix
            };
        }
    }

    public class ConvolutionalLayer : INeuralLayer
    {
        private int[] _inputDimensions;
        private int[] _outputDimensions;
        private IRegularization _regularization;
        private IActivation _activation;
        private int _numOfFilter;
        private Matrix[] _filters;
        private Matrix _input;
        private Matrix[] _biases;
        private Matrix _output;
        private int[] _filterOutputDimensions;
        private Matrix[] _filterOutputs;
        private Matrix[] _deltaWeights;
        private Matrix[] _gradients;

        public IRegularization Regularization
        {
            get
            {
                return this._regularization;
            }
            set
            {
                this._regularization = value;
            }
        }

        public IActivation Activation
        {
            get
            {
                return this._activation;
            }
            set
            {
                this._activation = value;
            }
        }

        public IList<double> FlatData
        {
            get
            {
                List<double> flatData = new List<double>();

                for (int filterIndex = 0; filterIndex < this.NumOfFilters; filterIndex++)
                {
                    flatData.AddRange(this.Filters[filterIndex].GetFlatData());
                    flatData.AddRange(this.Biases[filterIndex].GetFlatData());
                }

                return flatData;
            }
            set
            {
                int index = 0;

                for (int filterIndex = 0; filterIndex < this.NumOfFilters; filterIndex++)
                {
                    this.Filters[filterIndex].Map(() => value[index++]);
                    this.Biases[filterIndex].Map(() => value[index++]);
                }
            }
        }

        public int[] InputDimensions
        {
            get
            {
                return this._inputDimensions;
            }
            private set
            {
                this._inputDimensions = value;
            }
        }

        public int[] OutputDimensions
        {
            get
            {
                return this._outputDimensions;
            }
            private set
            {
                this._outputDimensions = value;
            }
        }

        public Matrix[] Biases
        {
            get
            {
                return this._biases;
            }
            private set
            {
                this._biases = value;
            }
        }

        public int NumOfFilters
        {
            get
            {
                return this._numOfFilter;
            }
            private set
            {
                this._numOfFilter = value;
            }
        }

        public Matrix[] Filters
        {
            get
            {
                return this._filters;
            }
            private set
            {
                this._filters = value;
            }
        }

        public Matrix Input
        {
            get
            {
                return this._input;
            }
            private set
            {
                this._input = value;
            }
        }

        public Matrix Outputs
        {
            get
            {
                return this._output;
            }
            private set
            {
                this._output = value;
            }
        }

        public Matrix[] Gradients
        {
            get
            {
                return this._gradients;
            }
            private set
            {
                this._gradients = value;
            }
        }

        public Matrix[] DeltaWeights
        {
            get
            {
                return this._deltaWeights;
            }
            private set
            {
                this._deltaWeights = value;
            }
        }

        public ConvolutionalLayer(int[] inputDimension, int[] filterDimensions, int numOfFilters)
        {
            this.InputDimensions = inputDimension;
            this.NumOfFilters = numOfFilters;

            this._filterOutputDimensions = Matrix.GenerateInfoForCrossCorrelation(this.InputDimensions, filterDimensions);

            this.Filters = new Matrix[this.NumOfFilters];
            this.Biases = new Matrix[this.NumOfFilters];
            this._filterOutputs = new Matrix[this.NumOfFilters];
            this.DeltaWeights = new Matrix[this.NumOfFilters];
            this.Gradients = new Matrix[this.NumOfFilters];

            for (int filterIndex = 0; filterIndex < this.NumOfFilters; filterIndex++)
            {
                Matrix filter = new Matrix(filterDimensions);
                this.Filters[filterIndex] = filter;

                Matrix bias = new Matrix(this._filterOutputDimensions);
                this.Biases[filterIndex] = bias;
            }
            int[] outputDimensions = new int[] { this.NumOfFilters }.Concat(this._filterOutputDimensions).ToArray();
            this.OutputDimensions = Matrix.SimplifyDimensions(outputDimensions);
            this.Randomize();
        }

        public ConvolutionalLayer Randomize()
        {
            for (int filterIndex = 0; filterIndex < this.NumOfFilters; filterIndex++)
            {
                this.Filters[filterIndex].Map(() => ThreadSafeRandom.Gaussian(0, 1.0 / Math.Sqrt(this.Filters[filterIndex].Data.Length)));
                this.Biases[filterIndex].Map(() => ThreadSafeRandom.Gaussian());
            }

            return this;
        }

        public Matrix FeedForward(Matrix inputs)
        {
            this.Input = inputs.Clone() as Matrix;
            List<double> flatOutput = new List<double>();

            for (int filterIndex = 0; filterIndex < this.NumOfFilters; filterIndex++)
            {
                Matrix filterOutput = inputs % this.Filters[filterIndex];
                filterOutput += this.Biases[filterIndex];
                filterOutput.Map((double value) => this.Activation.Function(value));

                flatOutput.AddRange(filterOutput.GetFlatData());

                this._filterOutputs[filterIndex] = filterOutput;
            }

            this.Outputs = new Matrix(this.OutputDimensions).PopulateFromFlatData(flatOutput.ToArray());

            return this.Outputs.Clone() as Matrix;
        }

        public Matrix BackPropagation(Matrix error)
        {
            Matrix propagatedErrors = new Matrix(this.InputDimensions);
            for (int filterIndex = 0; filterIndex < this.NumOfFilters; filterIndex++)
            {
                Matrix filterError = new Matrix(this._filterOutputDimensions);
                filterError.PopulateFromFlatData(error.Data.Skip(filterIndex * filterError.Data.Length).Take(filterError.Data.Length).ToArray());

                Matrix filterGradient = (this._filterOutputs[filterIndex].Clone() as Matrix).Map((double val) => this.Activation.Derivative(val));
                filterGradient ^= filterError;
                this.Gradients[filterIndex] = filterGradient;

                this.DeltaWeights[filterIndex] = this.Input % filterGradient;

                Matrix propEr = Matrix.PrepareMatrixForFullConvolution(filterGradient, this.Filters[filterIndex]) % this.Filters[filterIndex];
                propagatedErrors += propEr;
            }

            return propagatedErrors;
        }

        public void CorrectWeights(double learningRate, double regularizationRate)
        {
            for (int filterIndex = 0; filterIndex < this.NumOfFilters; filterIndex++)
            {
                this.Biases[filterIndex] -= (this.Gradients[filterIndex] * learningRate);

                Matrix regularization = (this.Filters[filterIndex].Clone() as Matrix).Map((double value) =>
                {
                    this.Regularization.Function(value);
                    return this.Regularization.Derivative(value);
                });

                this.Filters[filterIndex] -= (this.DeltaWeights[filterIndex] + (regularization * regularizationRate)) * learningRate;
            }
        }

        public object Clone()
        {
            throw new NotImplementedException();
        }
    }

    /*
    /// <summary>
    /// The basic building block of the <see cref="NeuralNetworkOld"/> class.
    /// </summary>
    [Serializable]
    public class FullyConnectedLayerOld : INeuralLayer
    {
        //private int _numberOfInputs;
        //private int _numberOfOutputs;
        //private double[] _inputs;
        //private double[] _outputs;
        //private double[,] _weights;
        //private double[,] _deltaWeights;
        //private double[] _propagatedError;
        private IActivation _activation;
        private IRegularization _regularization;
        private Matrix _weights;
        private Matrix _error;
        private Matrix _deltaWeight;

        ///// <summary>
        ///// Gets a number of input neurons in current layer including bias.
        ///// </summary>
        //public int NumberOfInputs
        //{
        //    get
        //    {
        //        return this._numberOfInputs;
        //    }
        //    private set
        //    {
        //        this._numberOfInputs = value;
        //    }
        //}

        ///// <summary>
        ///// Gets a number of output neurons in current layer.
        ///// </summary>
        //public int NumberOfOutputs
        //{
        //    get
        //    {
        //        return this._numberOfOutputs;
        //    }
        //    private set
        //    {
        //        this._numberOfOutputs = value;
        //    }
        //}

        ///// <summary>
        ///// A single-dimensional array of <see cref="double"/>'s to hold input values.
        ///// </summary>
        //public double[] Inputs
        //{
        //    get
        //    {
        //        return this._inputs;
        //    }
        //    private set
        //    {
        //        this._inputs = value;
        //    }
        //}

        ///// <summary>
        ///// A single-dimensional array of <see cref="double"/>'s to hold output values.
        ///// </summary>
        //public double[] Outputs
        //{
        //    get
        //    {
        //        return this._outputs;
        //    }
        //    private set
        //    {
        //        this._outputs = value;
        //    }
        //}

        ///// <summary>
        ///// A two-dimensional array of <see cref="double"/>'s to hold the weights of the connections between each input and output neurons.
        ///// </summary>
        //public double[,] Weights
        //{
        //    get
        //    {
        //        return this._weights;
        //    }
        //    private set
        //    {
        //        this._weights = value;
        //    }
        //}

        ///// <summary>
        ///// A two-dimensional array of <see cref="double"/>'s containing a change by which each weight should be changed to minimize the error.
        ///// </summary>
        //public double[,] DeltaWeights
        //{
        //    get
        //    {
        //        return this._deltaWeights;
        //    }
        //    private set
        //    {
        //        this._deltaWeights = value;
        //    }
        //}

        ///// <summary>
        ///// A single-dimensional array of <see cref="double"/>'s containing a cumulative gradient of the error from most outer layer down to current layer by which the Delta of each weight should be calculated.
        ///// </summary>
        //public double[] PropagatedError
        //{
        //    get
        //    {
        //        return this._propagatedError;
        //    }
        //    private set
        //    {
        //        this._propagatedError = value;
        //    }
        //}

        public IActivation Activation
        {
            get
            {
                return this._activation;
            }
            set
            {
                if (value == null)
                {
                    value = new Identity();
                }
                this._activation = value;
            }
        }

        public IRegularization Regularization
        {
            get
            {
                return this._regularization;
            }
            set
            {
                if (value == null)
                {
                    value = new NoRegularization();
                }
                this._regularization = value;
            }
        }

        public Matrix Weights
        {
            get
            {
                return this._weights;
            }

            private set
            {
                this._weights = value;
            }
        }

        public Matrix Error
        {
            get
            {
                return this._error;
            }

            private set
            {
                this._error = value;
            }
        }

        public Matrix DeltaWeights
        {
            get
            {
                return this._deltaWeight;
            }
            private set
            {
                this._deltaWeight = value;
            }
        }

        private Matrix _biases;

        public Matrix Biases
        {
            get
            {
                return this._biases;
            }
            set
            {
                this._biases = value;
            }
        }

        private Matrix _inputs;

        public Matrix Inputs
        {
            get
            {
                return this._inputs;
            }
            set
            {
                this._inputs = value;
            }
        }

        private Matrix _outputs;
        private readonly Matrix _propagatedError;

        public Matrix Outputs
        {
            get
            {
                return this._outputs;
            }
            set
            {
                this._outputs = value;
            }
        }

        public Matrix PropagatedError
        {
            get
            {
                return this._propagatedError;
            }
        }

        public FullyConnectedLayerOld(int numOfInputs, int numOfOutputs)
        {
            this.Weights = new Matrix(numOfInputs, numOfOutputs);
            this.Error = new Matrix(numOfOutputs);
            this.DeltaWeights = new Matrix(numOfInputs, numOfOutputs);

            RandomizeWeights();
        }

        public Matrix FeedForward(Matrix inputs)
        {
            this.Inputs = inputs.Clone() as Matrix;
            this.Outputs = this.Weights * this.Inputs;
            this.Outputs += this.Biases;
            this.Outputs.Map(this.Activation.Function);
            return this.Outputs;
        }

        public void BackPropagation(Matrix error)
        {
        }

        public void BackPropagation(Matrix propagatedErrorOuter, Matrix weightsOuter)
        {
            throw new NotImplementedException();
        }

        public void RandomizeWeights()
        {
            double totalInputs = this.Weights.Columns;
            this.Weights.Map(() => ThreadSafeRandom.Gaussian(0.0, 1.0 / Math.Sqrt(totalInputs)));
        }

        public void CorrectWeights(double LearningRate, double regularizationRate)
        {
            throw new NotImplementedException();
        }

        public object Clone()
        {
            throw new NotImplementedException();
        }


        ///// <summary>
        ///// Initializes a new instance of the <see cref="NeuralLayers"/> class.
        ///// </summary>
        ///// <param name="activationFunction">One of the <see cref="Utilities.ActivationFunction"/>.</param>
        ///// <param name="NumberOfInputs">Number of input neurons without a bias.</param>
        ///// <param name="NumberOfOutputs">Number of output neurons.</param>
        //public FullyConnectedLayer(IActivation activationFunction, int NumberOfInputs, int NumberOfOutputs)
        //{
        //    this.Activation = activationFunction;

        //    this.NumberOfInputs = NumberOfInputs + 1; //Adds Bias to input neurons and saves it to a local field.
        //    this.NumberOfOutputs = NumberOfOutputs;   //Saves number of output neurons to a local field.

        //    this.Inputs = new double[this.NumberOfInputs];    //Initializes Inputs array with correct capacity.
        //    this.Outputs = new double[this.NumberOfOutputs];  //Initializes Outputs array with correct capacity.
        //    this.Weights = new double[this.NumberOfOutputs, this.NumberOfInputs];  //Initializes Weights array with correct capacity. For each output - there are 'n' number of input connections.
        //    this.DeltaWeights = new double[this.NumberOfOutputs, this.NumberOfInputs]; //Initializes DeltaWeights array with correct capacity. For each output - there are 'n' number of input connections.
        //    this.PropagatedError = new double[this.NumberOfOutputs];  //Initializes CumulativeDelta array with correct capacity.

        //    RandomizeWeights(); //Randomazies all weights to be from -0.5 to 0.5.

        //    this.Regularization = new NoRegularization();
        //}

        ///// <summary>
        ///// Propagates inputs throught weights and activation function to get the output.
        ///// </summary>
        ///// <param name="_Inputs">The input array of <see cref="double"/>'s to be propagated forward. The size must be the same as specified in <see cref="NeuralLayers"/>'s constructor (without bias).</param>
        ///// <returns>The product of Weights by Inputs matrices passed through activation function.</returns>
        //public double[] FeedForward(double[] _Inputs)
        //{
        //    for (int inputIndex = 0; inputIndex < this.NumberOfInputs - 1; inputIndex++) //Copies all input values from _Input to a local field, except last local elements that is bias.
        //    {
        //        this.Inputs[inputIndex] = _Inputs[inputIndex];
        //    }

        //    this.Inputs[this.NumberOfInputs - 1] = 1.0; //Sets the value of bias to 1.

        //    for (int outputIndex = 0; outputIndex < this.NumberOfOutputs; outputIndex++) //For each output neuron:
        //    {
        //        this.Outputs[outputIndex] = 0;   //Resets the output value of the neuron.

        //        for (int inputIndex = 0; inputIndex < this.NumberOfInputs; inputIndex++) //For each connection to input neuron:
        //        {
        //            this.Outputs[outputIndex] += this.Inputs[inputIndex] * this.Weights[outputIndex, inputIndex];  //Adds inputs multiplied by corresponding weight to the output neuron.
        //        }

        //        this.Outputs[outputIndex] = this.Activation.Function(this.Outputs[outputIndex]);    //Applies activation function to each output neuron.
        //    }

        //    return this.Outputs; //Returns calculated product of Weights by Inputs matrices with applied activation function.
        //}

        ///// <summary>
        ///// Calculates DeltaWeights for output neurons only.
        ///// </summary>
        ///// <param name="Error">The single-dimensional array of <see cref="double"/>'s representing a cost, or error, value for each output neuron.</param>
        //public void BackPropagation(double[] Error)
        //{
        //    for (int outputIndex = 0; outputIndex < this.NumberOfOutputs; outputIndex++) //For each output neuron:
        //    {
        //        this.PropagatedError[outputIndex] = Error[outputIndex] * this.Activation.Derivative(this.Outputs[outputIndex]); //Sets a PropagatedError of each output neuron to be the error of that neuron multiplied by derivative of activation function at that particular output (based on Chain Rule, derivative).

        //        for (int inputIndex = 0; inputIndex < this.NumberOfInputs; inputIndex++) //For each connection to input neuron:
        //        {
        //            this.DeltaWeights[outputIndex, inputIndex] = this.PropagatedError[outputIndex] * this.Inputs[inputIndex];  //Sets DeltaWeight of each connection (weight) to be a product of CumulativeDelta/Error multiplied by Input value (based on gradient).
        //        }
        //    }
        //}

        ///// <summary>
        ///// Calculates DeltaWeight for hidden and/or input neurons only.
        ///// </summary>
        ///// <param name="PropagatedErrorOuter">A single-dimensional array of <see cref="double"/>'s representing PropagatedError from next outer layer, that must be calculated before.</param>
        ///// <param name="WeightsOuter">A two-dimensional array of <see cref="double"/>'s representing all weights (connections) values from net outer layer.</param>
        //public void BackPropagation(double[] PropagatedErrorOuter, double[,] WeightsOuter)
        //{
        //    for (int outputIndex = 0; outputIndex < this.NumberOfOutputs; outputIndex++) //For each output neuron:
        //    {
        //        this.PropagatedError[outputIndex] = 0;   //Reset the local PropagatedError to calculate new one.

        //        for (int cdfIndex = 0; cdfIndex < PropagatedErrorOuter.Length; cdfIndex++)    //For each PropagatedError in next outer layer:
        //        {
        //            this.PropagatedError[outputIndex] += PropagatedErrorOuter[cdfIndex] * WeightsOuter[cdfIndex, outputIndex];   //Adds a product of next outer layer's PropagatedError by the Weight of the connection.
        //        }

        //        this.PropagatedError[outputIndex] *= this.Activation.Derivative(this.Outputs[outputIndex]); //Based on the Chain Rule, PropagatedError by the derivative of activation function for that particular output.

        //        for (int inputIndex = 0; inputIndex < this.NumberOfInputs; inputIndex++) //For each connection to input neuron:
        //        {
        //            this.DeltaWeights[outputIndex, inputIndex] = this.PropagatedError[outputIndex] * this.Inputs[inputIndex];  //Sets DeltaWeight of each connection (weight) to be a product of PropagatedError multiplied by Input value (based on gradient).
        //        }
        //    }
        //}

        ///// <summary>
        ///// Changes each weight/connection to minimize error.
        ///// </summary>
        ///// <param name="LearningRate">The magnitude by which each weight is to be changed.</param>
        ///// <param name="regularizationRate">The rate by wight L2 regularization occures. L2 regularization minimizes weights, so that all of them are close to 0.0.</param>
        //public void CorrectWeights(double LearningRate, double regularizationRate)
        //{
        //    this.Regularization.Reset();

        //    for (int outputIndex = 0; outputIndex < this.NumberOfOutputs; outputIndex++) //For each output neuron:
        //    {
        //        for (int inputIndex = 0; inputIndex < this.NumberOfInputs - 1; inputIndex++) //For each connection to input neuron:
        //        {
        //            this.Regularization.Function(this.Weights[outputIndex, inputIndex]);
        //            //this.Weights[outputIndex, inputIndex] = (1.0 - regularizationRate) * this.Weights[outputIndex, inputIndex];   //L2 regularization is proportional to weight itself, so each weight that is not connected to bies is reduced by some portion determined by regularizationRate.
        //            this.Weights[outputIndex, inputIndex] -= ((this.DeltaWeights[outputIndex, inputIndex] + (regularizationRate * this.Regularization.Derivative(this.Weights[outputIndex, inputIndex]))) * LearningRate); //Adds a negative gradient (derivative) to the weight approach a local minimum of Error/Cost function. The magnitude of change is determined by LearningRate.
        //        }

        //        this.Weights[outputIndex, this.NumberOfInputs - 1] -= (this.DeltaWeights[outputIndex, this.NumberOfInputs - 1] * LearningRate); //Adds a negative gradient (derivative) to the weight approach a local minimum of Error/Cost function. The magnitude of change is determined by LearningRate.
        //    }
        //}

        ///// <summary>
        ///// Randomizes all weights based on the number of inputs.
        ///// </summary>
        //public void RandomizeWeights()
        //{
        //    for (int outputIndex = 0; outputIndex < this.NumberOfOutputs; outputIndex++) //For each output neuron:
        //    {
        //        for (int inputIndex = 0; inputIndex < this.NumberOfInputs; inputIndex++) //For each connection to input neuron:
        //        {
        //            this.Weights[outputIndex, inputIndex] = ThreadSafeRandom.Gaussian(0.0, 1.0 / (Math.Sqrt(this.NumberOfInputs))); //Gets and saves a random double value ((from 0.0 up to 1.0) - 0.5 => (from -0.5 up to 0.5)) to a particular connection/weight. Cross-thread interference is prevented using ThreadSafeRandom class.
        //        }
        //    }
        //}

        ///// <summary>
        ///// Returns a deep copy of the <see cref="NeuralLayers"/> object.
        ///// </summary>
        ///// <returns>A deep copy of the <see cref="NeuralLayers"/> object.</returns>
        //public object Clone()
        //{
        //    FullyConnectedLayer copied = new FullyConnectedLayer(this.Activation, this.NumberOfInputs - 1, this.NumberOfOutputs) //initializes a new NueralLayer object with all fields being deep copied to new object.
        //    {
        //        Inputs = this.Inputs.Clone() as double[],
        //        Outputs = this.Outputs.Clone() as double[],
        //        Weights = this.Weights.Clone() as double[,],
        //        DeltaWeights = this.DeltaWeights.Clone() as double[,],
        //        PropagatedError = this.PropagatedError.Clone() as double[],
        //        Activation = this.Activation.Clone() as IActivation,
        //        Regularization = this.Regularization.Clone() as IRegularization
        //    };

        //    return copied;  //Returns a new deep copy of this object.
        //}
    }
    */
}
