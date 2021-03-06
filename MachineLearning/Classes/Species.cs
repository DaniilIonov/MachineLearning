﻿using System.Collections.Generic;

namespace MachineLearning
{
    public class Species : ISpecies
    {
        private double _fitness;
        private NeuralNetwork _brain;

        public double Fitness
        {
            get
            {
                return this._fitness;
            }
            set
            {
                this._fitness = value;
            }
        }

        public IList<double> DNA
        {
            get
            {
                return this.Brain.GetFlatData();
            }

            set
            {
                this.Brain.SetFromFlatData(value);
            }
        }

        public NeuralNetwork Brain
        {
            get
            {
                return this._brain;
            }
            set
            {
                this._brain = value;
            }
        }

        public Species()
        {
            this.Fitness = 0.0;
            this.Brain = new NeuralNetwork();
        }

        public Species(params int[] layersInfo)
        {
            this.Fitness = 0.0;
            this.Brain = new NeuralNetwork(layersInfo);
        }

        public Species(params int[][] layersInfo)
        {
            this.Fitness = 0.0;
            this.Brain = new NeuralNetwork(layersInfo);
        }

        public object Clone()
        {
            return new Species()
            {
                Fitness = this.Fitness,
                Brain = this.Brain.Clone() as NeuralNetwork
            };
        }

        public override string ToString()
        {
            return string.Format("The Fitness value is {0}", this.Fitness);
        }
    }
    /*
    /// <summary>
    /// Represents a derived <see cref="NeuralNetworkOld"/> with a <see cref="Species.Fitness"/> to be used in <see cref="Population{T}"/>, where T is derived from <see cref="Species"/>.
    /// </summary>
    [Serializable]
    public class Species : NeuralNetworkOld, ICloneable, ISpecies
    {
        private double _fitness;

        /// <summary>
        /// Represents overall performance of <see cref="Species"/>. Higher the fitness, higher the chances of this <see cref="Species"/> to breed.
        /// </summary>
        public double Fitness
        {
            get
            {
                return this._fitness;
            }
            set
            {
                this._fitness = value;
            }
        }

        /// <summary>
        /// Represents a sinle-dimensional array/list of <see cref="double"/>'s, containing all weights of <see cref="NeuralNetworkOld"/> in sequence.
        /// </summary>
        public List<double> DNA
        {
            get
            {
                return GetWeightsAsList();
            }
            set
            {
                SetWeightsFromList(value);
            }
        }

        /// <summary>
        /// Initializes an instance of <see cref="Species"/> with the specified parameters.
        /// </summary>
        /// <param name="activationFunction">One of the <see cref="Utilities.ActivationFunction"/>.</param>
        /// <param name="layersInfo">A list of number of neurons in each layer starting at input layer. Can be entered as an array of <see cref="int"/>'s, or comma-separated <see cref="int"/> values.</param>
        public Species(int firstLayer, int secondLayer, params int[] restInfo) : base(firstLayer, secondLayer, restInfo)
        {
            this.Fitness = 0.0;  //Resets fitness.
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="NeuralNetworkOld"/> with specified number of neurons in each layer, <see cref="NeuralLayers.Weights"/> are going to be populated from specified DNA parameter.
        /// </summary>
        /// <param name="activationFunction">One of the <see cref="Utilities.ActivationFunction"/>.</param>
        /// <param name="layersInfo">A list of number of neurons in each layer starting at input layer. Can be entered as array of <see cref="int"/>'s, or comma-separated <see cref="int"/> values.</param>
        /// <param name="_DNA">A <see cref="List{T}"/> of Weights, where T is <see cref="double"/>, to populate all <see cref="NeuralLayers.Weights"/>.</param>
        public Species(int[] layersInfo, List<double> _DNA) : base(layersInfo[0], layersInfo[1], layersInfo.ToList().GetRange(2, layersInfo.Length - 2).ToArray())
        {
            this.DNA = _DNA; //Populates weights from given list.
            this.Fitness = 0.0;  //Resets fitness.
        }

        /// <summary>
        /// Sets all weights in <see cref="NeuralNetworkOld"/> from a <see cref="List{T}"/>, where T is <see cref="double"/>. Can be used to initialize weights from a single-dimensional array of <see cref="double"/>'s.
        /// </summary>
        /// <param name="InputWeights"></param>
        public void SetWeightsFromList(List<double> InputWeights)
        {
            int index = 0;  //Resets the index that will be used for single-dimensional array.

            foreach (FullyConnectedLayer Layer in this.Layers)   //For each FullyConnectedLayer in Layers list:
            {
                //for (int outputIndex = 0; outputIndex < Layer.NumberOfOutputs; outputIndex++)   //For each output neuron in that Layer:
                //{
                //    for (int inputIndex = 0; inputIndex < Layer.NumberOfInputs; inputIndex++)   //For each connection/weight to a particular input nueron in that Layer:
                //    {
                //        Layer.Weights[outputIndex, inputIndex] = InputWeights[index];           //Set the value in that connection/weight from a single-dimensional array.
                //        index++;                                                                //Increment index by 1 to be used in the next iteration of the loop.
                //    }
                //}
            }
        }

        /// <summary>
        /// Returns a single-dimesional array of <see cref="double"/>'s, representing all weights in <see cref="NeuralNetworkOld"/>.
        /// </summary>
        /// <returns>Return a <see cref="List{T}"/>, where T is <see cref="double"/>. Contains </returns>
        public List<double> GetWeightsAsList()
        {
            List<double> WeightsList = new List<double>();  //Initializes empty list of double's to hold all weight.

            foreach (FullyConnectedLayer Layer in this.Layers)   //For each FullyConnectedLayer in Layers list:
            {
                //for (int outputIndex = 0; outputIndex < Layer.NumberOfOutputs; outputIndex++)   //For each output neuron in that Layer:
                //{
                //    for (int inputIndex = 0; inputIndex < Layer.NumberOfInputs; inputIndex++)   //For each connection/weight to a particular input nueron in that Layer:
                //    {
                //        WeightsList.Add(Layer.Weights[outputIndex, inputIndex]);    //Adds this connection/weight value to new list.
                //    }
                //}
            }

            return WeightsList; //Returns populated list.
        }

        /// <summary>
        /// Return a string containing a formated fitness value.
        /// </summary>
        /// <returns>A string containing a formated fitness value.</returns>
        public override string ToString()
        {
            return string.Format("The Fitness value is {0}", this.Fitness);
        }

        /// <summary>
        /// Creates a deep copy of the <see cref="Species"/> object.
        /// </summary>
        /// <returns>A copy of original <see cref="Species"/> object, populated with the same data.</returns>
        public override object Clone()
        {
            Species copied = new Species(this.LayersInfo.ToArray(), this.DNA)
            {
                Fitness = this.Fitness
            };

            return copied;
        }
    }
    */
}
