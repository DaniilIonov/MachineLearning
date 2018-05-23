using System;
using System.Collections.Generic;
using System.Linq;

namespace MachineLearning
{
    /// <summary>
    /// Represents a collection of <see cref="Species"/>. Contains methods and properties to evolve array of <see cref="Species"/> to maximize the <see cref="Species.Fitness"/> value.
    /// </summary>
    /// <typeparam name="T">Type must be derived from <see cref="Species"/> class. The constructor of type T must have the same parameters as a constructor of <see cref="Species"/>.</typeparam>
    [Obsolete("The class in currently under development")]
    [Serializable]
    public class Population<T> : IPopulation<T> where T : Species
    {
        private List<T> _members;
        private int _size;
        private int[] _layersInfo;
        private T _bestMember;
        private int _currentGeneration;
        private Utilities.CrossoverType crossoverType;
        private Utilities.CrossoverDelegate _crossover;
        private Utilities.MutationType mutationType;
        private Utilities.MutationDelegate _mutation;
        private double _mutationRate;
        private double _maxMutationMagnitude;
        private double _percentToKeep;
        private bool _isSorted;
        private bool _isNormalized;
        private bool _isFiltered;
        private IActivation _activation;

        /// <summary>
        /// Represents a list of all <see cref="Species"/> objects, contained in <see cref="Population{T}"/> class, where T is derived from <see cref="Species"/>.
        /// </summary>
        public List<T> Members
        {
            get
            {
                return this._members;
            }
            private set
            {
                this._members = value;
            }
        }

        /// <summary>
        /// Represents the number of <see cref="Species"/> in <see cref="Population{T}"/>.
        /// </summary>
        public int Size
        {
            get
            {
                return this._size;
            }
            private set
            {
                this._size = value;
            }
        }

        /// <summary>
        /// Contains information about number of neurons in each layer, starting from input layer, ending with output layer.
        /// </summary>
        public int[] LayersInfo
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
        /// Represents a deep copy of the <see cref="Species"/> with the highest numberic value of <see cref="Species.Fitness"/> before each breeding cycle.
        /// </summary>
        public T BestMember
        {
            get
            {
                return this._bestMember;
            }
            private set
            {
                this._bestMember = value;
            }
        }

        /// <summary>
        /// Represents a current total number of full breeding cycles past from the creation of <see cref="Population{T}"/>.
        /// </summary>
        public int CurrentGeneration
        {
            get
            {
                return this._currentGeneration;
            }
            private set
            {
                this._currentGeneration = value;
            }
        }

        /// <summary>
        /// Represents the type of crossover to be used. One of <see cref="Utilities.CrossoverType"/>
        /// </summary>
        public Utilities.CrossoverType CrossoverType
        {
            get
            {
                return this.crossoverType;
            }
            set
            {
                switch (value)
                {
                    case Utilities.CrossoverType.SinglePoint:
                        this.Crossover = Utilities.SinglePointCrossover;
                        break;
                    case Utilities.CrossoverType.Uniform:
                        this.Crossover = Utilities.UniformCrossover;
                        break;
                    default:
                        break;
                }
                this.crossoverType = value;
            }
        }

        /// <summary>
        /// A crossover function to be used to create children genotypes.
        /// </summary>
        public Utilities.CrossoverDelegate Crossover
        {
            get
            {
                return this._crossover;
            }
            private set
            {
                this._crossover = value;
            }
        }

        /// <summary>
        /// Represents a mutation type. One of <see cref="Utilities.MutationType"/>.
        /// </summary>
        public Utilities.MutationType MutationType
        {
            get
            {
                return this.mutationType;
            }
            set
            {
                switch (value)
                {
                    case Utilities.MutationType.Uniform:
                        this.Mutation = Utilities.UniformMutation;
                        break;
                    case Utilities.MutationType.Gaussian:
                        this.Mutation = Utilities.GaussianMutation;
                        break;
                    default:
                        break;
                }
                this.mutationType = value;
            }
        }


        /// <summary>
        /// A mutation function to alter child genome.
        /// </summary>
        public Utilities.MutationDelegate Mutation
        {
            get
            {
                return this._mutation;
            }
            private set
            {
                this._mutation = value;
            }
        }

        /// <summary>
        /// The double-precision floaing-poing number represening a percent chance of each Weight to mutate and get a random value. Default value is 0.1.
        /// </summary>
        public double MutationRate
        {
            get
            {
                return this._mutationRate;
            }
            set
            {
                this._mutationRate = value;
            }
        }

        /// <summary>
        /// The double-precision floaing-poing number represening the magnitude of change in the Wieght in percentage of Weight value itself. Default value is 0.5.
        /// </summary>
        public double MaxMutationMagnitude
        {
            get
            {
                return this._maxMutationMagnitude;
            }
            set
            {
                this._maxMutationMagnitude = value;
            }
        }

        /// <summary>
        /// The double-precision floaing-poing number represening the percentage of <see cref="Population{T}.Members"/> to keep for breeding. Default value is 0.5.
        /// </summary>
        public double PercentToKeep
        {
            get
            {
                return this._percentToKeep;
            }
            set
            {
                this._percentToKeep = value;
            }
        }

        /// <summary>
        /// A boolean read-only property that indicates if current population is sorted by fitness. Set to false after breeding.
        /// </summary>
        public bool IsSorted
        {
            get
            {
                return this._isSorted;
            }
            private set
            {
                this._isSorted = value;
            }
        }

        /// <summary>
        /// A boolean read-only property that indicates if all fitness values of the current population are normilzed relative to each other. Set to false after breeding.
        /// </summary>
        public bool IsNormalized
        {
            get
            {
                return this._isNormalized;
            }
            private set
            {
                this._isNormalized = value;
            }
        }

        /// <summary>
        /// A boolean read-only property that indicates if worst performing species are filtered out from the current population. Set to false after breeding.
        /// </summary>
        public bool IsFiltered
        {
            get
            {
                return this._isFiltered;
            }
            private set
            {
                this._isFiltered = value;
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
                this._activation = value;
            }
        }


        /// <summary>
        /// Initializes a new instance of <see cref="Population{T}"/>, where T is derived from <see cref="Species"/>, with indicated <see cref="Population{T}.Size"/> and structure of <see cref="NeuralNetworkOld"/>.
        /// </summary>
        /// <param name="Size">The number of <see cref="Species"/> that the new <see cref="Population{T}"/> will have.</param>
        /// <param name="activationFunction">One of the <see cref="Utilities.ActivationFunction"/>.</param>
        /// <param name="LayersInfo">A list of number of neurons in each layer starting at input layer. Can be entered as an array of <see cref="int"/>'s, or comma-separated <see cref="int"/> values.</param>
        public Population(int Size, params int[] LayersInfo)
        {
            this.Size = Size;
            this.LayersInfo = LayersInfo;
            this.Activation = new Identity();
            this.CurrentGeneration = 1; //Resets the number of generations.
            this.MutationRate = 0.1; //Default value is 0.1.
            this.MaxMutationMagnitude = 0.5; //Defauld magnitude of mutation is +-50%.
            this.PercentToKeep = 0.5; //By default, only best performing half of population is breeding.

            this.Members = new List<T>();    //Initializes an emty list of Species.
            this.BestMember = Activator.CreateInstance(typeof(T), new object[] { LayersInfo[0], LayersInfo[1], LayersInfo.ToList().GetRange(2, LayersInfo.Length - 2).ToArray() }) as T;    //Initializes an instance of <T> with the same parameters as Species constructor.

            this.CrossoverType = Utilities.CrossoverType.SinglePoint; //Initializes default crossover type.
            this.MutationType = Utilities.MutationType.Gaussian;

            for (int speciesIndex = 0; speciesIndex < this.Size; speciesIndex++)    //Initializes the number of T Species specified in Size parameter.
            {
                this.Members.Add(Activator.CreateInstance(typeof(T), new object[] { LayersInfo[0], LayersInfo[1], LayersInfo.ToList().GetRange(2, LayersInfo.Length - 2).ToArray() }) as T);    //Initializes an instance of <T> with the same parameters as Species constructor.
            }
        }

        /// <summary>
        /// Perfors an ascending bubble-sort algorythm based on <see cref="Species.Fitness"/>.
        /// </summary>
        public void SortSpeciesByFitness()
        {
            T temp; //Creates a blank Species reference.

            for (int i = 0; i < this.Members.Count; i++) //For all Species:
            {
                for (int j = 0; j < this.Members.Count - 1; j++) //Check for all remaining Species:
                {
                    if (this.Members[j].Fitness > this.Members[j + 1].Fitness)    //If current Fitness is greater that Fitness of the next Species - swap their positions, so that Species with the highest Fitness end up at the end of List.
                    {
                        temp = this.Members[j];
                        this.Members[j] = this.Members[j + 1];
                        this.Members[j + 1] = temp;
                    }
                }
            }

            this.BestMember = this.Members.Last().Clone() as T;   //Creates a deep copy of best performing Species to a local field.

            this.IsSorted = true;
        }

        /// <summary>
        /// Normalizes all <see cref="Species.Fitness"/> values to be relaive to minimum and maximum <see cref="Species.Fitness"/> values of current <see cref="Population{T}"/> using <see cref="Utilities.Map(double, double, double, double, double)"/> method.
        /// </summary>
        public void NormalizeFitness()
        {
            double minFitness = this.Members.First().Fitness;    //Saves the minimum fitness of whole population.
            double maxFitness = this.Members.Last().Fitness; //Saves the maximum fitness of whole population.

            if (minFitness == maxFitness)
            {
                minFitness = maxFitness - 1.0;
            }

            double sum = 0;
            foreach (T Member in this.Members)
            {
                Member.Fitness = Utilities.Map(Member.Fitness, minFitness, maxFitness, 0, 1);
                sum += Member.Fitness;
            }

            foreach (T Member in this.Members)
            {
                Member.Fitness *= (100.0 / sum);
            }

            this.IsNormalized = true;
        }

        /// <summary>
        /// Performs a probability based random selection of the <see cref="Species"/> based on <see cref="Species.Fitness"/>. Higher the <see cref="Species.Fitness"/> - higher the chance of <see cref="Species"/> to be selected. All <see cref="Species"/> in <see cref="Population{T}"/> must be sorted.
        /// </summary>
        /// <param name="ascending">Represends ascending or descending order of evaluation. True - higher <see cref="Species.Fitness"/> value means higher chance of been chosen, False - lower value of <see cref="Species.Fitness"/> means higher chance of been chosen.</param>
        /// <returns>A randomly chosen <see cref="Species"/> based on its <see cref="Species.Fitness"/> value.</returns>
        public T GetRandomSpecies(bool ascending = true)
        {
            //If descending is specified, then Species list is reversed, for Fitness values to be inverted later on.
            if (!ascending)
            {
                this.Members.Reverse();
            }

            double cumSum = 0.0;    //Resets the total sum of all Fitness values.
            foreach (Species Member in this.Members) //For each Species in Population:
            {
                double fitnessToAdd = ascending ? Member.Fitness : Utilities.Map(Member.Fitness, 0.0, 100.0, 100.0, 0.0);   //If Ascending is specified, then the Fitness remains unchanged. However, when descending is specified, the Fitness value is inverted from 0.0 -> 100.0 and from 100.0 -> 0.0;
                cumSum += fitnessToAdd;   //Increment total sum by the value of the Species fitness.
            }

            double randomProb = ThreadSafeRandom.NextDouble() * cumSum; //Gets the random double value from 0.0 up to total Fitness, representing a random cumulative fitness of a particular Species.
            double targetSum = 0;   //Resets the target fitness sum.

            T chosenSpecies = null; //Creates a placeholder for return value.
            foreach (Species Member in this.Members) //For each Species in Population:
            {
                double fitnessToCheck = ascending ? Member.Fitness : Utilities.Map(Member.Fitness, 0.0, 100.0, 100.0, 0.0);   //If Ascending is specified, then the Fitness remains unchanged. However, when descending is specified, the Fitness value is inverted from 0.0 -> 100.0 and from 100.0 -> 0.0;
                if (randomProb <= (targetSum += fitnessToCheck))    //First increments target sum by Fitness value, then compares new target sum to random cumulative fitness.
                {
                    chosenSpecies = Member as T; //Returns chosen Species when reaches the target random cumulative fitness.
                    break;
                }
            }

            //If descending is specified, then Species list is reversed, for Fitness values to be inverted later on.
            if (!ascending)
            {
                this.Members.Reverse();
            }

            return chosenSpecies;   //Returns whatever Species has been chosen.
        }

        /// <summary>
        /// Mutates all <see cref="Species"/> in <see cref="Population{T}"/> based on MutationRate.
        /// </summary>
        public void MutateSpecies()
        {
            foreach (Species Member in this.Members) //For each Species in Population:
            {
                Member.DNA = this.Mutation(this as IPopulation<ISpecies>, Member.DNA.ToList()); //Changes the Weight of the Species by calling the MutateBrain method, that returns modified Weights list.
            }
        }

        /// <summary>
        /// Main aspect of evolutionary algorythm. <see cref="Species"/> with higher <see cref="Species.Fitness"/> has higher change of passing its <see cref="Species.DNA"/> to offsprings.
        /// </summary>
        /// <returns>A number representing highest <see cref="Species.Fitness"/> value among all <see cref="Population{T}.Members"/>.</returns>
        public double ToBreed()
        {
            if (!this.IsSorted)
            {
                SortSpeciesByFitness(); //Sorts Species in ascending order.
            }
            double bestFitness = this.BestMember.Fitness;    //Saves the fitnes of Best Species to be return from function.
            if (!this.IsNormalized)
            {
                NormalizeFitness(); //Mormalizes all Species Fitness values relative to minimum and maximum Fitness among Population.
            }

            if (!this.IsFiltered)
            {
                FilterByPerformance();
            }

            List<T> newPopulation = new List<T>();  //Initializes an empty List for population of new Speceies.
            newPopulation.AddRange(this.Members);

            while (newPopulation.Count < this.Size)  //Keeps going untill newPopulation has reached the target Population Size.
            {
                //Gets a ramdom Species with probability proportional to its Fitness.
                Species parentA = GetRandomSpecies();
                Species parentB = GetRandomSpecies();

                //Saves DNA of each parent Species.
                List<double> parentA_DNA = parentA.DNA.ToList();
                List<double> parentB_DNA = parentB.DNA.ToList();

                //Calls CreateChildren function to get new children from parents genome.
                List<double>[] children = this.Crossover(parentA_DNA, parentB_DNA);
                //Saves new child genome in local variable.
                List<double> child1_DNA = children[0];
                List<double> child2_DNA = children[1];

                //Mutates Weights of new children/Species with specified mutationRate chance.
                child1_DNA = this.Mutation(this as IPopulation<ISpecies>, child1_DNA);
                child2_DNA = this.Mutation(this as IPopulation<ISpecies>, child2_DNA);

                //Appends new instances of Species based on DNA of newly creaded children/Species to the newPopulation.
                newPopulation.Add(Activator.CreateInstance(typeof(T), new object[] { this.LayersInfo[0], this.LayersInfo[1], this.LayersInfo.ToList().GetRange(2, this.LayersInfo.Length - 2).ToArray(), child1_DNA }) as T);
                newPopulation.Add(Activator.CreateInstance(typeof(T), new object[] { this.LayersInfo[0], this.LayersInfo[1], this.LayersInfo.ToList().GetRange(2, this.LayersInfo.Length - 2).ToArray(), child2_DNA }) as T);
            }

            newPopulation = newPopulation.GetRange(0, this.Size);    //Constraints the size of newPopulatio to be exactly the targer Population Size.

            this.Members.Clear();
            this.Members.InsertRange(0, newPopulation);
            ResetFitness();
            //Members = newPopulation;    //Replaces old Members with new ones.

            this.CurrentGeneration++;   //Increments number of generations by 1.

            this.IsSorted = false;
            this.IsNormalized = false;
            this.IsFiltered = false;

            return bestFitness; //Returns best fitness from before breeding cycle.
        }

        /// <summary>
        /// Filters out the worst performing species from the popualtion.
        /// </summary>
        public void FilterByPerformance()
        {
            while (this.Members.Count > (this.Size * this.PercentToKeep) && this.Members.Count > 2)    //Kills worst performing half of population based on their Fitness.
            {
                //Members.Remove(GetRandomSpecies(false));    //Removes probabily based chosen Species from List of Members.
                this.Members.Remove(this.Members.First());    //Removes probabily based chosen Species from List of Members.
            }

            this.IsFiltered = true;
        }

        /// <summary>
        /// Sets all values of fitness to be 0.
        /// </summary>
        public void ResetFitness()
        {
            foreach (T Member in this.Members)
            {
                Member.Fitness = 0;
            }
        }

        ///// <summary>
        ///// Trains each <see cref="Species"/> to approach given correct output(s) based on input(s).
        ///// </summary>
        ///// <param name="Inputs">A single-dimensional array of <see cref="double"/>'s at which weights adjustments to be performed.</param>
        ///// <param name="CorrectOutputs">A single-dimensional array of <see cref="double"/>'s representing correct set of outputs for the given inputs.</param>
        ///// <param name="LearningRate">The magnitude by which each weight is to be changed.</param>
        ///// <param name="regulregularizationRate">The rate by wight L2 regularization occures. L2 regularization minimizes weights, so that all of them are close to 0.0.</param>
        //public void TrainSpecies(double[] Inputs, double[] CorrectOutputs, double LearningRate, double regulregularizationRate)
        //{
        //    foreach (Species Member in this.Members) //For each Species in Population:
        //    {
        //        Member.Train(Inputs, CorrectOutputs, LearningRate, regulregularizationRate);    //Calls the Train() method of NeuralNetwork.
        //    }
        //}

        ///// <summary>
        ///// Trains each <see cref="Species"/> to approach given correct input-output pairs from <see cref="IOList"/> parameter, with indicated batch size.
        ///// </summary>
        ///// <param name="IOSets">A list of input and correct output pairs.</param>
        ///// <param name="batchSize">Number of input-output pairs to be processed at once to improve generalization.</param>
        ///// <param name="LearningRate">The magnitude by which each weight is to be changed.</param>
        ///// <param name="regulregularizationRate">The rate by wight L2 regularization occures. L2 regularization minimizes weights, so that all of them are close to 0.0.</param>
        //public void TrainSpecies(IOList IOSets, int batchSize, double LearningRate, double regulregularizationRate)
        //{
        //    foreach (Species Member in this.Members)
        //    {
        //        Member.Train(IOSets, batchSize, LearningRate, regulregularizationRate);    //Calls the Train() method of NeuralNetwork.
        //    }
        //}

        /// <summary>
        /// Returns a string contining the <see cref="Population{T}.Size"/>.
        /// </summary>
        /// <returns>A string contining the <see cref="Population{T}.Size"/>.</returns>
        public override string ToString()
        {
            return $"The population size is {this.Size}";
        }
    }
}
