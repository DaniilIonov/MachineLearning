using System.Collections.Generic;

namespace MachineLearning
{
    /// <summary>
    /// Defines a Population Interface to be used with Neuroevolution.
    /// </summary>
    public interface IPopulation<T> where T : ISpecies
    {
        /// <summary>
        /// Represents a list of all species in current population.
        /// </summary>
        List<T> Members { get; }

        /// <summary>
        /// Representds the total number of species in current population.
        /// </summary>
        int Size { get; }

        /// <summary>
        /// Represents the best performing species from previous generation.
        /// </summary>
        T BestMember { get; }

        /// <summary>
        /// Represents current generation.
        /// </summary>
        int CurrentGeneration { get; }

        /// <summary>
        /// The double-precision floaing-poing number represening a percent chance of each Weight to mutate and get a random value. Default value is 0.1.
        /// </summary>
        double MutationRate { get; set; }

        /// <summary>
        /// The double-precision floaing-poing number represening the magnitude of change in the Wieght in percentage of Weight value itself. Default value is 0.5.
        /// </summary>
        double MaxMutationMagnitude { get; set; }

        /// <summary>
        /// The double-precision floaing-poing number represening the percentage of <see cref="Population{T}.Members"/> to keep for breeding. Default value is 0.5.
        /// </summary>
        double PercentToKeep { get; set; }

        /// <summary>
        /// Represents a mutation type. One of <see cref="Utilities.MutationType"/>.
        /// </summary>
        Utilities.MutationType MutationType { get; set; }

        /// <summary>
        /// Represents the type of crossover to be used. One of <see cref="Utilities.CrossoverType"/>
        /// </summary>
        Utilities.CrossoverType CrossoverType { get; set; }

        /// <summary>
        /// A crossover function to be used to create children genotypes.
        /// </summary>
        Utilities.CrossoverDelegate Crossover { get; }

        /// <summary>
        /// A mutation method to alter child genome.
        /// </summary>
        Utilities.MutationDelegate Mutation { get; }
    }
}
