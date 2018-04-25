﻿using System.Collections.Generic;

namespace MachineLearning
{
    /// <summary>
    /// Defines a species to be used to create a <see cref="Population{T}"/>, where T is derived from <see cref="Species"/>.
    /// </summary>
    public interface ISpecies
    {
        /// <summary>
        /// Represents overall performance of <see cref="Species"/>. Higher the fitness, higher the chances of this <see cref="Species"/> to breed.
        /// </summary>
        double Fitness { get; set; }

        /// <summary>
        /// Represents a sinle-dimensional array/list of <see cref="double"/>'s, containing all weights of <see cref="NeuralNetwork"/> in sequence.
        /// </summary>
        List<double> DNA { get; set; }
    }
}
