using System;
using System.Collections.Generic;
using System.Linq;

namespace MachineLearning
{
    /// <summary>
    /// Represents an artificial Neural Network made of multiple Neural Layers. Can be used as universal function approximator.
    /// </summary>
    [Serializable]
    public class NeuralNetwork : ICloneable
    {
        private int[][] _layersInfo;
        private List<INeuralLayer> _layers;
        private IRegularization _regularization;
        private Utilities.RegularizationType _regularizationType;
        private ICost _cost;
        private Utilities.CostType _costType;

        /// <summary>
        /// Contains information about the structure of neurons in each layer, starting from input layer, ending with output layer.
        /// </summary>
        public int[][] LayersInfo
        {
            get
            {
                return this._layersInfo;
            }
            private set
            {
                this._layersInfo = value;
            }
        }

        /// <summary>
        /// List of <see cref="INeuralLayer"/> that shapes <see cref="NeuralNetwork"/>.
        /// </summary>
        public List<INeuralLayer> Layers
        {
            get
            {
                return this._layers;
            }
            private set
            {
                this._layers = value;
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
                IRegularization regularization = value == null ? new NoRegularization() : value;
                foreach (INeuralLayer layer in this.Layers)
                {
                    layer.Regularization = regularization;
                }
                this._regularization = value;
            }
        }

        public ICost Cost
        {
            get
            {
                return this._cost;
            }
            set
            {
                this._cost = value;
            }
        }

        public Utilities.RegularizationType RegularizationType
        {
            get
            {
                return this._regularizationType;
            }
            set
            {
                switch (value)
                {
                    case Utilities.RegularizationType.L1:
                        this.Regularization = new L1Regularization();
                        break;
                    case Utilities.RegularizationType.L2:
                        this.Regularization = new L2Regularization();
                        break;
                    case Utilities.RegularizationType.None:
                        this.Regularization = new NoRegularization();
                        break;
                    default:
                        this.Regularization = new NoRegularization();
                        break;
                }

                this._regularizationType = value;
            }
        }

        public Utilities.CostType CostType
        {
            get
            {
                return this._costType;
            }
            set
            {
                switch (value)
                {
                    case Utilities.CostType.Quadratic:
                        this.Cost = new QuadraticCost();
                        break;
                    case Utilities.CostType.CrossEntropy:
                        this.Cost = new CrossEntropyCost();
                        break;
                    default:
                        this.Cost = new QuadraticCost();
                        break;
                }
                this._costType = value;
            }
        }


        public NeuralNetwork()
        {
            this.LayersInfo = new int[0][];
            InitializeLayers();
            this.CostType = Utilities.CostType.Quadratic;
            this.RegularizationType = Utilities.RegularizationType.None;
        }

        public NeuralNetwork(params int[] layersInfo)
        {
            if (layersInfo.Length < 2)
            {
                throw new ArgumentException("Nural Networks require at least input and output dimensions to be specified.", "layersInfo");
            }

            this.LayersInfo = new int[layersInfo.Length][];
            for (int layerIndex = 0; layerIndex < layersInfo.Length; layerIndex++)
            {
                this.LayersInfo[layerIndex] = new int[] { layersInfo[layerIndex], 1 };
            }

            InitializeLayers();
            this.CostType = Utilities.CostType.Quadratic;
            this.RegularizationType = Utilities.RegularizationType.None;
        }

        public NeuralNetwork(params int[][] layersInfo)
        {
            if (layersInfo.Length < 2)
            {
                throw new ArgumentException("Nural Networks require at least input and output dimensions to be specified.", "layersInfo");
            }

            this.LayersInfo = new int[layersInfo.Length][];
            for (int layerIndex = 0; layerIndex < layersInfo.Length; layerIndex++)
            {
                this.LayersInfo[layerIndex] = layersInfo[layerIndex].ToArray();
            }

            InitializeLayers();

            this.CostType = Utilities.CostType.Quadratic;
            this.RegularizationType = Utilities.RegularizationType.None;
        }

        private void InitializeLayers()
        {
            this.Layers = new List<INeuralLayer>();
            for (int layerIndex = 0; layerIndex < this.LayersInfo.Length - 1; layerIndex++)
            {
                this.Layers.Add(new FullyConnectedLayer(this.LayersInfo[layerIndex], this.LayersInfo[layerIndex + 1]).Randomize());
            }
        }

        public Matrix FeedForward(Matrix input)
        {
            Matrix output = this.Layers.First().FeedForward(input);

            for (int layerIndex = 1; layerIndex < this.Layers.Count; layerIndex++)
            {
                output = this.Layers[layerIndex].FeedForward(output);
            }

            return output;
        }

        public void BackPropagation(Matrix input, Matrix correctOutput, double learningRate, double regularizationRate = 0.0)
        {
            Matrix predictedOutput = this.FeedForward(input);

            this.Cost.Reset();
            Matrix error = new Matrix(predictedOutput.Dimensions).Map((int[] coords) =>
            {
                this.Cost.Function(predictedOutput[coords], correctOutput[coords]);
                return this.Cost.Derivative(predictedOutput[coords], correctOutput[coords]);
            });

            for (int layerIndex = this.Layers.Count - 1; layerIndex >= 0; layerIndex--)
            {
                error = this.Layers[layerIndex].BackPropagation(error);
            }

            this.Regularization.Reset();

            foreach (INeuralLayer layer in this.Layers)
            {
                layer.CorrectWeights(learningRate, regularizationRate);
            }
        }

        public List<double> GetFlatData()
        {
            List<double> data = new List<double>();

            foreach (INeuralLayer layer in this.Layers)
            {
                data.AddRange(layer.FlatData);
            }

            return data;
        }

        public void SetFromFlatData(IList<double> data)
        {
            int index = 0, numOfElements = 0;

            foreach (INeuralLayer layer in this.Layers)
            {
                numOfElements = layer.FlatData.Count;
                layer.FlatData = data.ToList().GetRange(index, numOfElements);
                index += numOfElements;
            }
        }

        public void AddInputLayer(INeuralLayer neuralLayer)
        {
            if (!this.LayersInfo.First().SequenceEqual(neuralLayer.OutputDimensions))
            {
                throw new ArgumentException("The ouput of neural layer does not match to the input of the neural network.", "neuralLayer");
            }

            int[][] newLayersInfo = new int[this.LayersInfo.Length + 1][];
            newLayersInfo[0] = neuralLayer.InputDimensions;
            for (int layerInfoIndex = 0; layerInfoIndex < this.LayersInfo.Length; layerInfoIndex++)
            {
                newLayersInfo[layerInfoIndex + 1] = this.LayersInfo[layerInfoIndex];
            }

            this.LayersInfo = newLayersInfo;
            this.Layers.Insert(0, neuralLayer);
        }

        public void AddOutputLayer(INeuralLayer neuralLayer)
        {
            if (!this.LayersInfo.Last().SequenceEqual(neuralLayer.InputDimensions))
            {
                throw new ArgumentException("The input of neural layer does not match to the output of the neural network.", "neuralLayer");
            }

            int[][] newLayersInfo = new int[this.LayersInfo.Length + 1][];
            newLayersInfo[this.LayersInfo.Length] = neuralLayer.OutputDimensions;
            for (int layerInfoIndex = 0; layerInfoIndex < this.LayersInfo.Length; layerInfoIndex++)
            {
                newLayersInfo[layerInfoIndex] = this.LayersInfo[layerInfoIndex];
            }

            this.LayersInfo = newLayersInfo;
            this.Layers.Add(neuralLayer);
        }

        public void AddInitialLayer(INeuralLayer neuralLayer)
        {
            int[][] newLayersInfo = new int[2][];
            newLayersInfo[0] = neuralLayer.InputDimensions;
            newLayersInfo[1] = neuralLayer.OutputDimensions;

            this.LayersInfo = newLayersInfo;
            this.Layers.Add(neuralLayer);
        }

        public object Clone()
        {
            List<INeuralLayer> newLayers = new List<INeuralLayer>();

            foreach (INeuralLayer layer in this.Layers)
            {
                newLayers.Add(layer.Clone() as INeuralLayer);
            }

            NeuralNetwork newNeuralNetwork = new NeuralNetwork(this.LayersInfo)
            {
                Layers = newLayers,
                CostType = this.CostType,
                RegularizationType = this.RegularizationType
            };

            return newNeuralNetwork;
        }
    }
    /*
    /// <summary>
    /// Represents an artificial Neural Network made of multiple Neural Layers. Can be used as universal function approximator.
    /// </summary>
    [Serializable]
    public class NeuralNetworkOld : ICloneable
    {
        private List<int> _layersInfo;
        private List<INeuralLayer> _layers;
        private Queue<double> _errorQueue;
        private IActivation _activation;
        private ICost _cost;
        private IRegularization _regularization;

        /// <summary>
        /// Contains information about number of neurons in each layer, starting from input layer, ending with output layer.
        /// </summary>
        public List<int> LayersInfo
        {
            get
            {
                return this._layersInfo;
            }
            private set
            {
                this._layersInfo = value;
            }
        }

        /// <summary>
        /// List of <see cref="NeuralLayers"/> (building blocks), contains information about <see cref="NeuralLayers.Weights"/>, <see cref="NeuralLayers.DeltaWeights"/> and <see cref="NeuralLayers.PropagatedError"/>.
        /// </summary>
        public List<INeuralLayer> Layers
        {
            get
            {
                return this._layers;
            }
            private set
            {
                this._layers = value;
            }
        }

        /// <summary>
        /// Contains last 100 error values. Can be used to determine if error is stabilized.
        /// </summary>
        public Queue<double> ErrorQueue
        {
            get
            {
                return this._errorQueue;
            }
            private set
            {
                this._errorQueue = value;
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
                if (value == null)
                {
                    value = new Identity();
                }
                foreach (INeuralLayer layer in this.Layers)
                {
                    layer.Activation = value;
                }
                this._activation = value;
            }
        }

        public ICost Cost
        {
            get
            {
                return this._cost;
            }
            set
            {
                if (value == null)
                {
                    value = new QuadraticCost();
                }
                this._cost = value;
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
                    value = new L1Regularization();
                }
                foreach (INeuralLayer layer in this.Layers)
                {
                    layer.Regularization = value;
                }
                this._regularization = value;
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="NeuralNetworkOld"/> with specified number of neurons in each layer.
        /// </summary>
        /// <param name="layersInfo">A list of number of neurons in each layer starting at input layer. Can be entered as an array of <see cref="int"/>'s, or comma-separated <see cref="int"/> values.</param>
        public NeuralNetworkOld(int firstLayer, int secondLayer, params int[] restInfo)
        {
            int[] layersInfo = new int[restInfo.Length + 2];
            layersInfo[0] = firstLayer;
            layersInfo[1] = secondLayer;
            for (int layerIndex = 2; layerIndex < restInfo.Length + 2; layerIndex++)
            {
                layersInfo[layerIndex] = restInfo[layerIndex - 2];
            }

            this.LayersInfo = layersInfo.ToList();   //Deep copies layersInfo to a local field.
            this.Layers = new List<INeuralLayer>();           //Initializes a List of NeuronLayer's.
            this.ErrorQueue = new Queue<double>();           //Initializes a Queue for error values.
            this.Activation = new Identity();            //Copies the type of activation function.
            this.Cost = new QuadraticCost();
            this.Regularization = new L1Regularization();

            for (int layerIndex = 0; layerIndex < this.LayersInfo.Count - 1; layerIndex++)  //For each pair of layer info, create a FullyConnectedLayer with specified number of inputs (without bias), and number of outputs that matches the number of inputs of the next layer.
            {
                //this.Layers.Add(new FullyConnectedLayer(this.LayersInfo[layerIndex], this.LayersInfo[layerIndex + 1]));  //Adds new instance of FullyConnectedLayer to a local List.
            }
        }

        public Matrix Guess(double[] Inputs)
        {
            this.Layers.First().FeedForward(new Matrix(Inputs)); //Manually feeds forward input array 

            for (int layerIndex = 1; layerIndex < this.Layers.Count; layerIndex++)   //For each layer starting at second one (because first one is propagated manually):
            {
                this.Layers[layerIndex].FeedForward(this.Layers[layerIndex - 1].Outputs);     //The output on previous layer goes to input of current layer.
            }

            return this.Layers.Last().Outputs;   //Returns the output of last layer, representing the output of the NeuralNetwork itself.
        }

        /// <summary>
        /// Trains the <see cref="NeuralNetworkOld"/> to approach given correct output(s) based on input(s).
        /// </summary>
        /// <param name="Inputs">A single-dimensional array of <see cref="double"/>'s at which weights adjustments to be performed.</param>
        /// <param name="CorrectOutputs">A single-dimensional array of <see cref="double"/>'s representing correct set of outputs for the given inputs.</param>
        /// <param name="LearningRate">The magnitude by which each weight is to be changed.</param>
        /// <param name="regularizationRate">The rate by wight L2 regularization occures. L2 regularization minimizes weights, so that all of them are close to 0.0.</param>
        /// <returns>A boolean value true if error is not stabilazed, or false if stable.</returns>
        public bool Train(double[] Inputs, double[] CorrectOutputs, double LearningRate, double regularizationRate)
        {
            double[] guessed = Guess(Inputs).Clone() as double[];   //Makes a deep copy of the output of the Guess function.

            double[] Error = new double[CorrectOutputs.Length];     //Creates an empty array of double's with the same size as output neurons.
            this.Cost.Reset();
            //this.TotalError = 0;                                         //Resets TotalError for current training set.
            //this.L2RegularizationPenalty = 0;

            for (int errorIndex = 0; errorIndex < Error.Length; errorIndex++)   //For each output neuron:
            {
                this.Cost.Function(guessed[errorIndex], CorrectOutputs[errorIndex]);
                Error[errorIndex] = this.Cost.Derivative(guessed[errorIndex], CorrectOutputs[errorIndex]);  //guessed[errorIndex] - CorrectOutputs[errorIndex];   //Error of each neuron is a difference of guessed output and correct output.
                //this.TotalError += (Error[errorIndex] * Error[errorIndex]);                  //Total error is ecrementer by the square of each neuron error.
            }

            //for (int layerIndex = 0; layerIndex < this.Layers.Count; layerIndex++)   //For each layer:
            //{
            //    for (int outputIndex = 0; outputIndex < this.Layers[layerIndex].NumberOfOutputs; outputIndex++)  //For each neuron in particular layer:
            //    {
            //        for (int inputIndex = 0; inputIndex < this.Layers[layerIndex].NumberOfInputs - 1; inputIndex++)  //For each connection to input neuron excluding connection to bias:
            //        {
            //            this.L2RegularizationPenalty += this.Layers[layerIndex].Weights[outputIndex, inputIndex] * this.Layers[layerIndex].Weights[outputIndex, inputIndex];    //Increments L2 penalty by square of the each weight.
            //        }
            //    }
            //}

            this.Layers.Last().BackPropagation(new Matrix(Error));   //Calls the Backpropagation algorithm for output layer of the Neural Network with calculated errors.

            for (int layerIndex = this.Layers.Count - 2; layerIndex >= 0; layerIndex--)  //For each layer of Neural Network starting at second last layer going to input layer:
            {
                this.Layers[layerIndex].BackPropagation(this.Layers[layerIndex + 1].PropagatedError, this.Layers[layerIndex + 1].Weights); //Calls Backpropagation algorythm for hidden and/or input neuron, passing PropagatedError and Weights from next outer layer.
            }

            for (int layerIndex = 0; layerIndex < this.Layers.Count; layerIndex++)   //For each layer in Neural Network:
            {
                this.Layers[layerIndex].CorrectWeights(LearningRate, regularizationRate);   //Correct Weights based on calculated DeltaWeights from Backpropagation algorythm.
            }

            ////Keeps the size of the Queue to be 100.
            //if (this.ErrorQueue.Count >= 100)    //If there are more than 100 elements in Queue:
            //{
            //    this.ErrorQueue.Dequeue();   //Releases oldest error from Queue.
            //}
            //this.ErrorQueue.Enqueue(this.TotalError); //Appends new error to the queue.

            //return ToTrain();    //Returns a boolean value true if error value is not stable yet; false if error is stabilazed.
            return false;
        }

        /// <summary>
        /// Trains the <see cref="NeuralNetworkOld"/> to approach given correct input-output pairs from <see cref="IOList"/> parameter, with indicated batch size.
        /// </summary>
        /// <param name="IOSets">A list of input and correct output pairs.</param>
        /// <param name="batchSize">Number of input-output pairs to be processed at once to improve generalization.</param>
        /// <param name="LearningRate">The magnitude by which each weight is to be changed.</param>
        /// <param name="regularizationRate">The rate by wight L2 regularization occures. L2 regularization minimizes weights, so that all of them are close to 0.0.</param>
        /// <returns>A boolean value true if error is not stabilazed, or false if stable.</returns>
        public bool Train(IOList IOSets, int batchSize, double LearningRate, double regularizationRate)
        {
            ////this.TotalError = 0; //Resets TotalError for current training set.
            ////this.L2RegularizationPenalty = 0; //Resets L2 regularization.

            ////To Safely divide IOSets into smaller batcher, 1st determine how many whole batches can fit is IOSets.
            //for (int batchNum = 0; batchNum < IOSets.Count / batchSize; batchNum++) //For whole number of batches in IOSets:
            //{
            //    double[] wholeBatchError = new double[this.LayersInfo.Last()];     //Creates an empty array of double's with the same size as number of output neurons.

            //    for (int batchIndex = 0; batchIndex < batchSize; batchIndex++)  //For number of input-output pairs in batch:
            //    {
            //        int ioIndex = batchNum + batchIndex * batchSize;    //The index of set can be calculated from formula index=x+y*width, where x is current number of the batch to be processed, y is currect element in batch, and width is the size of the bach.

            //        double[] Inputs = IOSets[ioIndex]["Inputs"].ToArray();  //Stores array of inputs into local variable from Dictionary.
            //        double[] CorrectOutputs = IOSets[ioIndex]["Outputs"].ToArray(); //Stores array of outputs into local variable from Dictionary.


            //        double[] guessed = Guess(Inputs).Clone() as double[];   //Makes a deep copy of the output of the Guess function.

            //        for (int errorIndex = 0; errorIndex < wholeBatchError.Length; errorIndex++)   //For each output neuron:
            //        {
            //            wholeBatchError[errorIndex] += ((guessed[errorIndex] - CorrectOutputs[errorIndex]) / batchSize);   //Error of each neuron is a difference of guessed output and correct output divided by the batchSize to get average.
            //            this.TotalError += (wholeBatchError[errorIndex] * wholeBatchError[errorIndex] / batchSize);                  //Total error is ecrementer by the square of each neuron error divided by batchSize to get average.
            //        }

            //        //for (int layerIndex = 0; layerIndex < this.Layers.Count; layerIndex++)   //For each layer:
            //        //{
            //        //    for (int outputIndex = 0; outputIndex < this.Layers[layerIndex].NumberOfOutputs; outputIndex++)  //For each neuron in particular layer:
            //        //    {
            //        //        for (int inputIndex = 0; inputIndex < this.Layers[layerIndex].NumberOfInputs - 1; inputIndex++)  //For each connection to input neuron excluding connection to bias:
            //        //        {
            //        //            this.L2RegularizationPenalty += (this.Layers[layerIndex].Weights[outputIndex, inputIndex] * this.Layers[layerIndex].Weights[outputIndex, inputIndex] / batchSize);    //Increments L2 regularization penalty by square of the each weight divided by batchSize to get average.
            //        //        }
            //        //    }
            //        //}
            //    }

            //    this.Layers.Last().BackPropagation(wholeBatchError);   //Calls the Backpropagation algorithm for output layer of the Neural Network with calculated errors.

            //    for (int layerIndex = this.Layers.Count - 2; layerIndex >= 0; layerIndex--)  //For each layer of Neural Network starting at second last layer going to input layer:
            //    {
            //        this.Layers[layerIndex].BackPropagation(this.Layers[layerIndex + 1].PropagatedError, this.Layers[layerIndex + 1].Weights); //Calls Backpropagation algorythm for hidden and/or input neuron, passing PropagatedError and Weights from next outer layer.
            //    }

            //    for (int layerIndex = 0; layerIndex < this.Layers.Count; layerIndex++)   //For each layer in Neural Network:
            //    {
            //        this.Layers[layerIndex].CorrectWeights(LearningRate, regularizationRate);   //Correct Weights based on calculated DeltaWeights from Backpropagation algorythm.
            //    }
            //}

            //double[] remainedBatchError = new double[this.LayersInfo.Last()];

            ////The remaining input-output pair that could not fit into whole-sized batches:
            //for (int batchIndex = 0; batchIndex < IOSets.Count % batchSize; batchIndex++)   //For each remained element:
            //{
            //    int ioIndex = (IOSets.Count / batchSize) * batchSize + batchIndex;  //The remained index can be calculated by adding remained batch index to the last index of whole-sized batch.

            //    double[] Inputs = IOSets[ioIndex]["Inputs"].ToArray();  //Stores array of inputs into local variable from Dictionary.
            //    double[] CorrectOutputs = IOSets[ioIndex]["Outputs"].ToArray(); //Stores array of outputs into local variable from Dictionary.


            //    double[] guessed = Guess(Inputs).Clone() as double[];   //Makes a deep copy of the output of the Guess function.

            //    this.TotalError = 0;                                         //Resets TotalError for current training set.

            //    for (int errorIndex = 0; errorIndex < remainedBatchError.Length; errorIndex++)   //For each output neuron:
            //    {
            //        remainedBatchError[errorIndex] = (guessed[errorIndex] - CorrectOutputs[errorIndex]) / (IOSets.Count % batchSize);   //Error of each neuron is a difference of guessed output and correct output divided by remained batchSize to get average.
            //        this.TotalError += (remainedBatchError[errorIndex] * remainedBatchError[errorIndex] / (IOSets.Count % batchSize));                  //Total error is ecrementer by the square of each neuron error divided by remained batchSize to get average.
            //    }

            //    for (int layerIndex = 0; layerIndex < this.Layers.Count; layerIndex++)   //For each layer:
            //    {
            //        for (int outputIndex = 0; outputIndex < this.Layers[layerIndex].NumberOfOutputs; outputIndex++)  //For each neuron in particular layer:
            //        {
            //            for (int inputIndex = 0; inputIndex < this.Layers[layerIndex].NumberOfInputs - 1; inputIndex++)  //For each connection to input neuron excluding connection to bias:
            //            {
            //                this.TotalError += (this.Layers[layerIndex].Weights[outputIndex, inputIndex] * this.Layers[layerIndex].Weights[outputIndex, inputIndex] / (IOSets.Count % batchSize));    //Increments total error by square of the each weight (L2 regression) divided by remained batchSize to get average.
            //            }
            //        }
            //    }
            //}

            //this.Layers.Last().BackPropagation(remainedBatchError);   //Calls the Backpropagation algorithm for output layer of the Neural Network with calculated errors.

            //for (int layerIndex = this.Layers.Count - 2; layerIndex >= 0; layerIndex--)  //For each layer of Neural Network starting at second last layer going to input layer:
            //{
            //    this.Layers[layerIndex].BackPropagation(this.Layers[layerIndex + 1].PropagatedError, this.Layers[layerIndex + 1].Weights); //Calls Backpropagation algorythm for hidden and/or input neuron, passing PropagatedError and Weights from next outer layer.
            //}

            //for (int layerIndex = 0; layerIndex < this.Layers.Count; layerIndex++)   //For each layer in Neural Network:
            //{
            //    this.Layers[layerIndex].CorrectWeights(LearningRate, regularizationRate);   //Correct Weights based on calculated DeltaWeights from Backpropagation algorythm.
            //}

            ////Keeps the size of the Queue to be 100.
            //if (this.ErrorQueue.Count >= 100)    //If there are more than 100 elements in Queue:
            //{
            //    this.ErrorQueue.Dequeue();   //Releases oldest error from Queue.
            //}
            //this.ErrorQueue.Enqueue(this.TotalError); //Appends new error to the queue.

            //return ToTrain();    //Returns a boolean value true if error value is not stable yet; false if error is stabilazed.

            return false;
        }

        /// <summary>
        /// Determines if <see cref="NeuralNetworkOld"/> has reached minimum error. Returns true if minimum is reached, and false is error is still changing.
        /// </summary>
        /// <returns>Returns true if minimum is reached, and false is error is still changing.</returns>
        private bool ToTrain()
        {
            return this.ErrorQueue.Average() == ((this.ErrorQueue.Min() + this.ErrorQueue.Max()) / 2.0) || true;   //If the average of all error values is equal to average of minimun and maximum error values.
        }

        /// <summary>
        /// Creates a deep copy of <see cref="NeuralNetworkOld"/> object.
        /// </summary>
        /// <returns>An objec containing a deep copy of the original <see cref="NeuralNetworkOld"/> object.</returns>
        public virtual object Clone()
        {
            NeuralNetworkOld copied = new NeuralNetworkOld(this.LayersInfo[0], this.LayersInfo[1], this.LayersInfo.GetRange(2, this.LayersInfo.Count - 2).ToArray())    //Creates a copy of the original NeuralNetwork with the same fields.
            {
                ErrorQueue = new Queue<double>(this.ErrorQueue),
                Activation = this.Activation.Clone() as IActivation,
                Cost = this.Cost.Clone() as ICost,
                Regularization = this.Regularization.Clone() as IRegularization
            };

            for (int lindex = 0; lindex < this.Layers.Count; lindex++)  //For each layer:
            {
                copied.Layers[lindex] = this.Layers[lindex].Clone() as INeuralLayer; //Stores a deep copy of FullyConnectedLayer into new object's Layers List.
            }

            return copied;  //Returns a deep copy.
        }

        /// <summary>
        /// Returns a <see cref="string"/> containing information about all layers in <see cref="NeuralNetworkOld"/>.
        /// </summary>
        /// <returns>A <see cref="string"/> representing information about <see cref="NeuralNetworkOld"/> with format: 'Layers info: {number of inputs}, ... , {number of outputs}.</returns>
        public override string ToString()
        {
            string str = "Layers info: ";
            foreach (int layerInfo in this.LayersInfo)   //layerInfo represents a number of neurons in particular layer.
            {
                str += layerInfo + " ";
            }
            return str;
        }

        /// <summary>
        /// Returns a formated formulas for each output neuron.
        /// </summary>
        /// <param name="labelsOnly">When set to true - returns formula wihout any coefficients, when false - returns formula with values of the weights from <see cref="NeuralLayers.Weights"/>.</param>
        /// <returns>Returns a <see cref="List{T}"/>, where T is <see cref="string"/>, representing a formula for each output neuron.</returns>
        public List<string> GetFormula(bool labelsOnly)
        {
            List<string> layerEquations = new List<string>();   //Initializes a list of strings to store formula for each neuron in first layer.

            List<string> formula = new List<string>();  //Initializes a list of strings to store formula for output neurons.

            if (labelsOnly)
            {
                for (int neuronIndex = 0; neuronIndex < this.LayersInfo[1]; neuronIndex++)   //For each neuron in he first hidden layer:
                {
                    layerEquations.Add("f(");   //Adds the beggining of activation function notation for neuron's formula.

                    for (int inputLayerIndex = 0; inputLayerIndex < this.LayersInfo[0]; inputLayerIndex++)   //For each connection/weight to input neurons:
                    {
                        layerEquations[neuronIndex] += $"W{0}{neuronIndex}{inputLayerIndex}*I{inputLayerIndex} + "; //Appends a labels representing a weight and input with format: W{layer index}{hidden neuron index}{input neuron index}.
                    }

                    layerEquations[neuronIndex] += $"W{0}{neuronIndex}b";   //Appends a connection/weight label to a bias in form: W{layer index}{hidden neurn index}{b - for the bias}.

                    layerEquations[neuronIndex] += ")"; //Closes the activation function.
                }
                formula = GetNextLayerFormula(layerEquations, 2, labelsOnly);  //Calls a function to bould the formula for all neurons in the NeuralNetwork and saves it into local formula list.
            }
            //If the values of the weights are requested:
            else
            {
                for (int neuronIndex = 0; neuronIndex < this.LayersInfo[1]; neuronIndex++)   //For each neuron in first hidden layer:
                {
                    layerEquations.Add("f(");   //Adds the beggining of activation function notation for neuron's formula.

                    for (int inputLayerIndex = 0; inputLayerIndex < this.LayersInfo[0]; inputLayerIndex++)   //For each connection/weight to input neurons:
                    {
                        //layerEquations[neuronIndex] += $"[{this.Layers[0].Weights[neuronIndex, inputLayerIndex]}]*I{inputLayerIndex} + ";    //Appends the value of the weight and input label with format: [weight value]*I{index of input neuron}.
                    }

                    //layerEquations[neuronIndex] += $"[{this.Layers[0].Weights[neuronIndex, this.LayersInfo[0]]}]";    //Appends a connection/weight value to a bias if form: [weight value].

                    layerEquations[neuronIndex] += ")"; //Closes the activation function.
                }
                formula = GetNextLayerFormula(layerEquations, 2, labelsOnly);  //Calls a function to bould the formula for all neurons in the NeuralNetwork and saves it into local formula list.
            }

            return formula;
        }

        /// <summary>
        /// Builds a formula for specified layer of the <see cref="NeuralNetworkOld"/> based on formulas of pervious layer's neurons.
        /// </summary>
        /// <param name="prevLayerEquations"></param>
        /// <param name="nextLayerIndex"></param>
        /// <param name="labelsOnly">When set to true - returns formula wihout any coefficients, when false - returns formula with values of the weights from <see cref="NeuralLayers.Weights"/>.</param>
        /// <returns>Returns a <see cref="List{T}"/>, where T is <see cref="string"/>, representing a formula for a specified <see cref="NeuralLayers"/>'s index.</returns>
        private List<string> GetNextLayerFormula(List<string> prevLayerEquations, int nextLayerIndex, bool labelsOnly)
        {
            if (nextLayerIndex < this.LayersInfo.Count) //if the requesed layer's formula if not an output layer:
            {
                if (labelsOnly)
                {
                    List<string> layerEquations = new List<string>();   //Initializes a list of strings to store formula for each neuron in first layer.

                    for (int neuronIndex = 0; neuronIndex < this.LayersInfo[nextLayerIndex]; neuronIndex++)   //For each neuron in the specified layer:
                    {
                        layerEquations.Add("f(");   //Adds the beggining of activation function notation for neuron's formula.

                        for (int prevLayerIndex = 0; prevLayerIndex < this.LayersInfo[nextLayerIndex - 1]; prevLayerIndex++)   //For each connection/weight to previous layer's neurons:
                        {
                            layerEquations[neuronIndex] += $"W{nextLayerIndex - 1}{neuronIndex}{prevLayerIndex}*{prevLayerEquations[prevLayerIndex]} + "; //Appends a labels representing a weight and input with format: W{layer index}{current layer's neuron index}{previous layer's neuron index}.
                        }

                        layerEquations[neuronIndex] += $"W{nextLayerIndex - 1}{neuronIndex}b";   //Appends a connection/weight label to a bias in form: W{layer index}{current layer's neurn index}{b - for the bias}.

                        layerEquations[neuronIndex] += ")"; //Closes the activation function.
                    }

                    return GetNextLayerFormula(layerEquations, nextLayerIndex + 1, labelsOnly); //Makes a recursive function call to itself to build a formula for next layers's neurons using formula for current layer as base.
                }
                //If the values of the weights are requested:
                else
                {
                    List<string> layerEquations = new List<string>();   //Initializes a list of strings to store formula for each neuron in first layer.

                    for (int neuronIndex = 0; neuronIndex < this.LayersInfo[nextLayerIndex]; neuronIndex++)   //For each neuron in specified layer:
                    {
                        layerEquations.Add("f(");   //Adds the beggining of activation function notation for neuron's formula.

                        for (int prevLayerIndex = 0; prevLayerIndex < this.LayersInfo[nextLayerIndex - 1]; prevLayerIndex++)   //For each connection/weight to the previous layer's neurons:
                        {
                            //layerEquations[neuronIndex] += $"[{this.Layers[nextLayerIndex - 1].Weights[neuronIndex, prevLayerIndex]}]*{prevLayerEquations[prevLayerIndex]} + ";    //Appends the value of the weight and the equation of previous layer's neuron with format: [weight value]*f(equation of previous layer's neuron).
                        }

                        //layerEquations[neuronIndex] += $"[{this.Layers[nextLayerIndex - 1].Weights[neuronIndex, this.LayersInfo[nextLayerIndex - 1]]}]";    //Appends a connection/weight value to a bias if form: [weight value].

                        layerEquations[neuronIndex] += ")"; //Closes the activation function.
                    }

                    return GetNextLayerFormula(layerEquations, nextLayerIndex + 1, labelsOnly); //Makes a recursive function call to itself to build a formula for next layers's neurons using formula for current layer as base.
                }
            }
            //If specified layer is output layer:
            else
            {
                for (int outputIndex = 0; outputIndex < prevLayerEquations.Count; outputIndex++)    //For each equation of the outout neuron:
                {
                    string header = $"O{outputIndex}("; //Initializes a header for the formula. Forms a function-like notation.
                    for (int inputIndex = 0; inputIndex < this.LayersInfo[0]; inputIndex++)  //For each input neuron:
                    {
                        header += $"I{inputIndex}"; //Appends a function parameter in form of I{input index}.
                        if (inputIndex < this.LayersInfo[0] - 1) //Parameters are separated by comma.
                        {
                            header += ", ";
                        }
                    }
                    header += ") = ";   //Closes an output nueron's function notation.

                    prevLayerEquations[outputIndex] = prevLayerEquations[outputIndex].Insert(0, header);    //Inserts a header at the beggining of the output neuron's formula.
                }

                return prevLayerEquations;  //Returns completed formulas for each output neuron.
            }
        }

        //public void AddInputLayer(INeuralLayer neuralLayer)
        //{
        //    this.Layers.Insert(0, neuralLayer.Clone() as INeuralLayer);
        //    this.LayersInfo.Insert(0, neuralLayer.NumberOfInputs - 1);
        //}

        //public void AddOutputLayer(INeuralLayer neuralLayer)
        //{
        //    this.Layers.Add(neuralLayer.Clone() as INeuralLayer);
        //    this.LayersInfo.Add(neuralLayer.NumberOfOutputs);
        //}

        //public void AddInitialLayer(INeuralLayer neuralLayer)
        //{
        //    this.Layers.Add(neuralLayer.Clone() as INeuralLayer);
        //    this.LayersInfo.Add(neuralLayer.NumberOfInputs - 1);
        //    this.LayersInfo.Add(neuralLayer.NumberOfOutputs);
        //}

        //public static NeuralNetwork operator +(NeuralNetwork neuralNetwork, INeuralLayer neuralLayer)
        //{
        //    NeuralNetwork newNN = neuralNetwork.Clone() as NeuralNetwork;
        //    newNN.AddInputLayer(neuralLayer);
        //    return newNN;
        //}

        //public static NeuralNetwork operator *(NeuralNetwork neuralNetwork, INeuralLayer neuralLayer)
        //{
        //    NeuralNetwork newNN = neuralNetwork.Clone() as NeuralNetwork;
        //    newNN.AddOutputLayer(neuralLayer);
        //    return newNN;
        //}

        //public static NeuralNetwork operator %(NeuralNetwork neuralNetwork, INeuralLayer neuralLayer)
        //{
        //    NeuralNetwork newNN = neuralNetwork.Clone() as NeuralNetwork;
        //    if (newNN.Layers.Count > 0)
        //    {
        //        newNN.AddInitialLayer(neuralLayer);
        //    }
        //    return newNN;
        //}
    }
    */
}
