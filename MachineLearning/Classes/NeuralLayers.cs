using System;

namespace MachineLearning
{
    /// <summary>
    /// The basic building block of the <see cref="NeuralNetwork"/> class.
    /// </summary>
    [Serializable]
    public class FullyConnectedLayer : ICloneable, INeuralLayer
    {
        /// <summary>
        /// Represents a method thar can be used as an activation function.
        /// </summary>
        public Utilities.ActivationFunctionDelegate ActivationFunction { get; private set; }

        /// <summary>
        /// Represents a method that can be used a derivative of activation function.
        /// </summary>
        public Utilities.ActivationFunctionDelegate ActivationFunctionDerivative { get; private set; }

        /// <summary>
        /// Gets a number of input neurons in current layer including bias.
        /// </summary>
        public int NumberOfInputs { get; private set; }

        /// <summary>
        /// Gets a number of output neurons in current layer.
        /// </summary>
        public int NumberOfOutputs { get; private set; }

        /// <summary>
        /// A single-dimensional array of <see cref="double"/>'s to hold input values.
        /// </summary>
        public double[] Inputs { get; private set; }

        /// <summary>
        /// A single-dimensional array of <see cref="double"/>'s to hold output values.
        /// </summary>
        public double[] Outputs { get; private set; }

        /// <summary>
        /// A two-dimensional array of <see cref="double"/>'s to hold the weights of the connections between each input and output neurons.
        /// </summary>
        public double[,] Weights { get; private set; }

        /// <summary>
        /// A two-dimensional array of <see cref="double"/>'s containing a change by which each weight should be changed to minimize the error.
        /// </summary>
        public double[,] DeltaWeights { get; private set; }

        /// <summary>
        /// A single-dimensional array of <see cref="double"/>'s containing a cumulative gradient of the error from most outer layer down to current layer by which the Delta of each weight should be calculated.
        /// </summary>
        public double[] PropagatedError { get; private set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="NeuralLayers"/> class.
        /// </summary>
        /// <param name="activationFunction">One of the non-linear <see cref="Functions.ActivationFunction"/>'s.</param>
        /// <param name="activationFunctionDerivative">Derivative of one of the non-linear <see cref="Functions.ActivationFunction"/>'s</param>
        /// <param name="NumberOfInputs">Number of input neurons without a bias.</param>
        /// <param name="NumberOfOutputs">Number of output neurons.</param>
        public FullyConnectedLayer(Utilities.ActivationFunctionDelegate activationFunction, Utilities.ActivationFunctionDelegate activationFunctionDerivative, int NumberOfInputs, int NumberOfOutputs)
        {
            ActivationFunction = activationFunction;                        //Saves activation function to a local field.
            ActivationFunctionDerivative = activationFunctionDerivative;    //Saves a derivative of activation function to a local field.

            this.NumberOfInputs = NumberOfInputs + 1; //Adds Bias to input neurons and saves it to a local field.
            this.NumberOfOutputs = NumberOfOutputs;   //Saves number of output neurons to a local field.

            Inputs = new double[this.NumberOfInputs];    //Initializes Inputs array with correct capacity.
            Outputs = new double[this.NumberOfOutputs];  //Initializes Outputs array with correct capacity.
            Weights = new double[this.NumberOfOutputs, this.NumberOfInputs];  //Initializes Weights array with correct capacity. For each output - there are 'n' number of input connections.
            DeltaWeights = new double[this.NumberOfOutputs, this.NumberOfInputs]; //Initializes DeltaWeights array with correct capacity. For each output - there are 'n' number of input connections.
            PropagatedError = new double[this.NumberOfOutputs];  //Initializes CumulativeDelta array with correct capacity.

            RandomizeWeights(); //Randomazies all weights to be from -0.5 to 0.5.
        }

        /// <summary>
        /// Propagates inputs throught weights and activation function to get the output.
        /// </summary>
        /// <param name="_Inputs">The input array of <see cref="double"/>'s to be propagated forward. The size must be the same as specified in <see cref="NeuralLayers"/>'s constructor (without bias).</param>
        /// <returns>The product of Weights by Inputs matrices passed through activation function.</returns>
        public double[] FeedForward(double[] _Inputs)
        {
            for (int inputIndex = 0; inputIndex < NumberOfInputs - 1; inputIndex++) //Copies all input values from _Input to a local field, except last local elements that is bias.
            {
                Inputs[inputIndex] = _Inputs[inputIndex];
            }

            Inputs[NumberOfInputs - 1] = 1.0; //Sets the value of bias to 1.

            for (int outputIndex = 0; outputIndex < NumberOfOutputs; outputIndex++) //For each output neuron:
            {
                Outputs[outputIndex] = 0;   //Resets the output value of the neuron.

                for (int inputIndex = 0; inputIndex < NumberOfInputs; inputIndex++) //For each connection to input neuron:
                {
                    Outputs[outputIndex] += Inputs[inputIndex] * Weights[outputIndex, inputIndex];  //Adds inputs multiplied by corresponding weight to the output neuron.
                }

                Outputs[outputIndex] = ActivationFunction(Outputs[outputIndex]);    //Applies activation function to each output neuron.
            }

            return Outputs; //Returns calculated product of Weights by Inputs matrices with applied activation function.
        }

        /// <summary>
        /// Calculates DeltaWeights for output neurons only.
        /// </summary>
        /// <param name="Error">The single-dimensional array of <see cref="double"/>'s representing a cost, or error, value for each output neuron.</param>
        public void BackPropagation(double[] Error)
        {
            for (int outputIndex = 0; outputIndex < NumberOfOutputs; outputIndex++) //For each output neuron:
            {
                PropagatedError[outputIndex] = Error[outputIndex] * ActivationFunctionDerivative(Outputs[outputIndex]); //Sets a PropagatedError of each output neuron to be the error of that neuron multiplied by derivative of activation function at that particular output (based on Chain Rule, derivative).

                for (int inputIndex = 0; inputIndex < NumberOfInputs; inputIndex++) //For each connection to input neuron:
                {
                    DeltaWeights[outputIndex, inputIndex] = PropagatedError[outputIndex] * Inputs[inputIndex];  //Sets DeltaWeight of each connection (weight) to be a product of CumulativeDelta/Error multiplied by Input value (based on gradient).
                }
            }
        }

        /// <summary>
        /// Calculates DeltaWeight for hidden and/or input neurons only.
        /// </summary>
        /// <param name="PropagatedErrorOuter">A single-dimensional array of <see cref="double"/>'s representing PropagatedError from next outer layer, that must be calculated before.</param>
        /// <param name="WeightsOuter">A two-dimensional array of <see cref="double"/>'s representing all weights (connections) values from net outer layer.</param>
        public void BackPropagation(double[] PropagatedErrorOuter, double[,] WeightsOuter)
        {
            for (int outputIndex = 0; outputIndex < NumberOfOutputs; outputIndex++) //For each output neuron:
            {
                PropagatedError[outputIndex] = 0;   //Reset the local PropagatedError to calculate new one.

                for (int cdfIndex = 0; cdfIndex < PropagatedErrorOuter.Length; cdfIndex++)    //For each PropagatedError in next outer layer:
                {
                    PropagatedError[outputIndex] += PropagatedErrorOuter[cdfIndex] * WeightsOuter[cdfIndex, outputIndex];   //Adds a product of next outer layer's PropagatedError by the Weight of the connection.
                }

                PropagatedError[outputIndex] *= ActivationFunctionDerivative(Outputs[outputIndex]); //Based on the Chain Rule, PropagatedError by the derivative of activation function for that particular output.

                for (int inputIndex = 0; inputIndex < NumberOfInputs; inputIndex++) //For each connection to input neuron:
                {
                    DeltaWeights[outputIndex, inputIndex] = PropagatedError[outputIndex] * Inputs[inputIndex];  //Sets DeltaWeight of each connection (weight) to be a product of PropagatedError multiplied by Input value (based on gradient).
                }
            }
        }

        /// <summary>
        /// Changes each weight/connection to minimize error.
        /// </summary>
        /// <param name="LearningRate">The magnitude by which each weight is to be changed.</param>
        /// <param name="regularizationRate">The rate by wight L2 regularization occures. L2 regularization minimizes weights, so that all of them are close to 0.0.</param>
        public void CorrectWeights(double LearningRate, double regularizationRate)
        {
            for (int outputIndex = 0; outputIndex < NumberOfOutputs; outputIndex++) //For each output neuron:
            {
                for (int inputIndex = 0; inputIndex < NumberOfInputs - 1; inputIndex++) //For each connection to input neuron:
                {
                    Weights[outputIndex, inputIndex] = (1.0 - regularizationRate) * Weights[outputIndex, inputIndex];   //L2 regularization is proportional to weight itself, so each weight that is not connected to bies is reduced by some portion determined by regularizationRate.
                    Weights[outputIndex, inputIndex] -= (DeltaWeights[outputIndex, inputIndex] * LearningRate); //Adds a negative gradient (derivative) to the weight approach a local minimum of Error/Cost function. The magnitude of change is determined by LearningRate.
                }

                Weights[outputIndex, NumberOfInputs - 1] -= (DeltaWeights[outputIndex, NumberOfInputs - 1] * LearningRate); //Adds a negative gradient (derivative) to the weight approach a local minimum of Error/Cost function. The magnitude of change is determined by LearningRate.
            }
        }

        /// <summary>
        /// Randomizes all weights to be from -0.5 up to 0.5.
        /// </summary>
        public void RandomizeWeights()
        {
            for (int outputIndex = 0; outputIndex < NumberOfOutputs; outputIndex++) //For each output neuron:
            {
                for (int inputIndex = 0; inputIndex < NumberOfInputs; inputIndex++) //For each connection to input neuron:
                {
                    Weights[outputIndex, inputIndex] = ThreadSafeRandom.NextDouble() - 0.5; //Gets and saves a random double value ((from 0.0 up to 1.0) - 0.5 => (from -0.5 up to 0.5)) to a particular connection/weight. Cross-thread interference is prevented using ThreadSafeRandom class.
                }
            }
        }

        /// <summary>
        /// Returns a deep copy of the <see cref="NeuralLayers"/> object.
        /// </summary>
        /// <returns>A deep copy of the <see cref="NeuralLayers"/> object.</returns>
        public object Clone()
        {
            FullyConnectedLayer copied = new FullyConnectedLayer(ActivationFunction, ActivationFunctionDerivative, NumberOfInputs - 1, NumberOfOutputs) //initializes a new NueralLayer object with all fields being deep copied to new object.
            {
                Inputs = this.Inputs.Clone() as double[],
                Outputs = this.Outputs.Clone() as double[],
                Weights = this.Weights.Clone() as double[,],
                DeltaWeights = this.DeltaWeights.Clone() as double[,],
                PropagatedError = this.PropagatedError.Clone() as double[]
            };

            return copied;  //Returns a new deep copy of this object.
        }
    }
}
