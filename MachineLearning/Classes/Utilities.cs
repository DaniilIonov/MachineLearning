using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace MachineLearning
{
    /// <summary>
    /// Utilities class contains ActivationFunction enumeration, ActivationFunctionDelegate and some additional methods to perform various mathematical operations. Also contains enumerations for evolutionary algorithms.
    /// </summary>
    [Serializable]
    public static class Utilities
    {
        public delegate double CostFunctionDelegate(double predicted, double correct);

        public enum RegularizationType
        {
            L1,
            L2,
            None
        }

        public enum CostType
        {
            Quadratic,
            CrossEntropy
        }

        /// <summary>
        /// Mutation alters one or more gene values in a <see cref="ISpecies.GetDNA()"/> from its initial state.
        /// </summary>
        public enum MutationType
        {
            /// <summary>
            /// This operator replaces the value of the chosen gene with a uniform random value selected between the user-specified upper and lower bounds for that gene.
            /// </summary>
            Uniform,

            /// <summary>
            /// This operator adds a unit Gaussian distributed random value to the chosen gene. If it falls outside of the user-specified lower or upper bounds for that gene, the new gene value is clipped.
            /// </summary>
            Gaussian
        }
        /// <summary>
        /// CrossoverType enumeration contains 2 major crossover types for evolutionary algorithms.
        /// </summary>
        public enum CrossoverType
        {
            /// <summary>
            /// A single crossover point on both parents' organism DNA is selected. All data beyond that point in either organism DNA is swapped between the two parent organisms.
            /// </summary>
            SinglePoint,

            /// <summary>
            /// The uniform crossover uses a fixed mixing ratio between two parents. The uniform crossover evaluates each bit in the parent DNA for exchange with a probability of 0.5.
            /// </summary>
            Uniform
        }

        /// <summary>
        /// ActivationFunction enumeration contains 4 major activation function types for neural networks.
        /// </summary>
        public enum ActivationFunction : byte
        {
            /// <summary>
            /// Suppresses value to fit within -1.0 to 1.0 range.
            /// </summary>
            TanH,

            /// <summary>
            /// Suppressed value to fit within 0.0 to 1.0 range.
            /// </summary>
            Logistic,

            /// <summary>
            /// ReLU activation function returns input itself if greater that 0.0, and 0.0 otherwise.
            /// </summary>
            ReLU,

            /// <summary>
            /// LeakyReLU activation function returns input if greater that 0.0, and 0.01 portion of input otherwise.
            /// </summary>
            LeakyReLU,

            Identity
        }

        /// <summary>
        /// Represents a method that will be used as activation function.
        /// </summary>
        /// <param name="input">Value to be non-linearly adjusted.</param>
        /// <returns>Adjusted input value.</returns>
        public delegate double ActivationFunctionDelegate(double input);

        /// <summary>
        /// Represents a method to be used as crossover in breeding algorithm.
        /// </summary>
        /// <param name="parentA_DNA">Genome of the first parent species.</param>
        /// <param name="parentB_DNA">Genotype of the second parent species.</param>
        /// <returns>An array of <see cref="List{T}"/> of <see cref="double"/>'s representing two new children genotypes.</returns>
        public delegate List<double>[] CrossoverDelegate(List<double> parentA_DNA, List<double> parentB_DNA);

        /// <summary>
        /// Uniform crossover has only one point of swapping of genes.
        /// </summary>
        /// <param name="parentA_DNA">Genome of the first parent species.</param>
        /// <param name="parentB_DNA">Genotype of the second parent species.</param>
        /// <returns>An array of <see cref="List{T}"/> of <see cref="double"/>'s representing two new children genotypes.</returns>
        public static List<double>[] SinglePointCrossover(List<double> parentA_DNA, List<double> parentB_DNA)
        {
            //Creates local array of children genomes.
            List<double>[] children = new List<double>[2];

            //Initializes an empty Lists to hold children DNA.
            List<double> child1 = new List<double>();
            List<double> child2 = new List<double>();

            int crossoverPoint = ThreadSafeRandom.Next(parentA_DNA.Count);  //Gets random index at which the swap occures.
                                                                            //Assigns first part of DNA from parents to each child.
            child1.AddRange(parentA_DNA.GetRange(0, crossoverPoint));
            child2.AddRange(parentB_DNA.GetRange(0, crossoverPoint));
            //Assigns secons half (after cross point) from parents to each child.
            child1.AddRange(parentB_DNA.GetRange(crossoverPoint, parentA_DNA.Count - crossoverPoint));
            child2.AddRange(parentA_DNA.GetRange(crossoverPoint, parentA_DNA.Count - crossoverPoint));

            //Assigns children genomes to the array values.
            children[0] = child1;
            children[1] = child2;

            return children;    //Returns array of 2 children genomes.
        }

        /// <summary>
        /// Each parent gene has a 50% chance of occuring in either first or second child.
        /// </summary>
        /// <param name="parentA_DNA">Genome of the first parent species.</param>
        /// <param name="parentB_DNA">Genotype of the second parent species.</param>
        /// <returns>An array of <see cref="List{T}"/> of <see cref="double"/>'s representing two new children genotypes.</returns>
        public static List<double>[] UniformCrossover(List<double> parentA_DNA, List<double> parentB_DNA)
        {
            //Creates local array of children genomes.
            List<double>[] children = new List<double>[2];

            //Initializes an empty Lists to hold children DNA.
            List<double> child1 = new List<double>();
            List<double> child2 = new List<double>();

            //For each gene in DNA:
            for (int geneIndex = 0; geneIndex < parentA_DNA.Count; geneIndex++)
            {
                //Random number will be less than 0.5 50% of time, like a coin toss (head or tail):
                if (ThreadSafeRandom.NextDouble() < 0.5)
                {
                    //If 'tail', assign gene from parent A to child 1, and from parent B to child 2.
                    child1.Add(parentA_DNA[geneIndex]);
                    child2.Add(parentB_DNA[geneIndex]);
                }
                else
                {
                    //If 'head', assign gene from parent B to child 1, and from parent A to child 2.
                    child1.Add(parentB_DNA[geneIndex]);
                    child2.Add(parentA_DNA[geneIndex]);
                }
            }

            //Assigns children genomes to the array values.
            children[0] = child1;
            children[1] = child2;

            return children;    //Returns array of 2 children genomes.
        }

        /// <summary>
        /// Represents a method to be used to alter child genome.
        /// </summary>
        /// <param name="population">A <see cref="IPopulation{T}"/> containing all information about mutation, such as <see cref="IPopulation{T}.MutationRate"/> and <see cref="IPopulation{T}.MaxMutationMagnitude"/></param>
        /// <param name="dna">A genome to alter.</param>
        /// <returns>A new <see cref="List{T}"/> of <see cref="double"/>'s representing an altered genome.</returns>
        public delegate List<double> MutationDelegate(IPopulation<ISpecies> population, List<double> dna);

        /// <summary>
        /// Each gene has a <see cref="IPopulation{T}.MutationRate"/> chance of getting new random value with the limits of <see cref="IPopulation{T}.MaxMutationMagnitude"/>
        /// </summary>
        /// <param name="population">A <see cref="IPopulation{T}"/> containing all information about mutation, such as <see cref="IPopulation{T}.MutationRate"/> and <see cref="IPopulation{T}.MutationType"/></param>
        /// <param name="dna">A genome to alter.</param>
        /// <returns>A new <see cref="List{T}"/> of <see cref="double"/>'s representing an altered genome.</returns>
        public static List<double> UniformMutation(IPopulation<ISpecies> population, List<double> dna)
        {
            List<double> localDNA = dna.ToList();

            for (int geneIndex = 0; geneIndex < localDNA.Count; geneIndex++)    //For each Weight in InputDNA list:
            {
                if (ThreadSafeRandom.NextDouble() < population.MutationRate)   //If the ramdomly generated double value is smaller that mutation rate => random value is from 0.0 to 1.0 representing 0-100%, and MutationRate is also a %-chance.
                {
                    localDNA[geneIndex] = Utilities.Map(ThreadSafeRandom.NextDouble(), 0.0, 1.0, -1.0 * population.MaxMutationMagnitude, population.MaxMutationMagnitude);
                }
            }

            return localDNA;
        }

        /// <summary>
        /// Each gene has a <see cref="IPopulation{T}.MutationRate"/> chance of adjusting its value the limits of <see cref="IPopulation{T}.MaxMutationMagnitude"/>
        /// </summary>
        /// <param name="population">A <see cref="IPopulation{T}"/> containing all information about mutation, such as <see cref="IPopulation{T}.MutationRate"/> and <see cref="IPopulation{T}.MutationType"/></param>
        /// <param name="dna">A genome to alter.</param>
        /// <returns>A new <see cref="List{T}"/> of <see cref="double"/>'s representing an altered genome.</returns>
        public static List<double> GaussianMutation(IPopulation<ISpecies> population, List<double> dna)
        {
            List<double> localDNA = dna.ToList();

            for (int geneIndex = 0; geneIndex < localDNA.Count; geneIndex++)    //For each Weight in InputDNA list:
            {
                if (ThreadSafeRandom.NextDouble() < population.MutationRate)   //If the ramdomly generated double value is smaller that mutation rate => random value is from 0.0 to 1.0 representing 0-100%, and MutationRate is also a %-chance.
                {
                    localDNA[geneIndex] += Utilities.Map(ThreadSafeRandom.NextDouble(), 0.0, 1.0, -1.0 * population.MaxMutationMagnitude, population.MaxMutationMagnitude) * localDNA[geneIndex];  //Adjusts current Weight.

                    localDNA[geneIndex] = Math.Min(Math.Max(localDNA[geneIndex], -1.5), 1.5);
                }
            }

            return localDNA;
        }

        /// <summary>
        /// An equation of a line in slope-intercept form: y=m*x+b. Default equation is y=x.
        /// </summary>
        /// <param name="X">Represents x coordinate.</param>
        /// <param name="A">Represents m coefficient.</param>
        /// <param name="B">Represents y intercept.</param>
        /// <returns>Y coordinate.</returns>
        public static double Line(double X, double A = 1, double B = 0)
        {
            return (A * X) + B;
        }

        /// <summary>
        /// An equation of a parabola in y=(a*x^2 + b*x + c) form.
        /// </summary>
        /// <param name="X">Represents x coordinate.</param>
        /// <param name="A">Represents 'a' coefficient.</param>
        /// <param name="B">Represents 'b' coefficient.</param>
        /// <param name="C">Represents 'c' term.</param>
        /// <returns>Y coordinate.</returns>
        public static double Parabola(double X, double A = 1, double B = 0, double C = 0)
        {
            return (A * X * X) + (B * X) + C;
        }

        /// <summary>
        /// An equation of hyperbola in y=(a*x^3 + b*x^2 + c*x + b) form.
        /// </summary>
        /// <param name="X">Represents x coordinate.</param>
        /// <param name="A">Represents 'a' coefficient.</param>
        /// <param name="B">Represents 'b' coefficient.</param>
        /// <param name="C">Represents 'c' coefficient.</param>
        /// <param name="D">Represents 'd' term.</param>
        /// <returns>Y coordinate.</returns>
        public static double Hyperbola(double X, double A = 1, double B = 0, double C = 0, double D = 0)
        {
            return (A * X * X * X) + (B * X * X) + (C * X) + D;
        }

        /// <summary>
        /// Re-maps a number from one range to another.
        /// </summary>
        /// <param name="X">Value to map.</param>
        /// <param name="fromSource">The lower bound of the value’s current range</param>
        /// <param name="toSource">The upper bound of the value’s current range</param>
        /// <param name="fromTarget">The lower bound of the value’s target range</param>
        /// <param name="toTarget">The upper bound of the value’s target range</param>
        /// <returns>Mapped value.</returns>
        public static double Map(double X, double fromSource, double toSource, double fromTarget, double toTarget)
        {
            return (X - fromSource) / (toSource - fromSource) * (toTarget - fromTarget) + fromTarget;
        }

        /// <summary>
        /// Suppresses value to fit within -1.0 to 1.0 range.
        /// </summary>
        /// <param name="X">Value to suppress.</param>
        /// <returns>Suppresed value.</returns>
        public static double Tanh(double X)
        {
            return Math.Tanh(X);
        }

        /// <summary>
        /// Derivative of <see cref="Tanh"/> function at the particular value. Can used to find gradient.
        /// </summary>
        /// <param name="X">Value at which the derivative is to be found.</param>
        /// <returns>Derivative of the <see cref="Tanh"/> function at the input value.</returns>
        public static double TanhDerivative(double X)
        {
            return 1 - (X * X);
        }

        /// <summary>
        /// Suppressed value to fit within 0.0 to 1.0 range.
        /// </summary>
        /// <param name="X">Value to suppress.</param>
        /// <returns>Suppressed value.</returns>
        public static double Logistic(double X)
        {
            return 1 / (1 + Math.Exp(-X));
        }

        /// <summary>
        /// Derivative of <see cref="Logistic"/> function at the particular value. Can used to find gradient.
        /// </summary>
        /// <param name="X">Value at which the derivative is to be found.</param>
        /// <returns>Derivative of the <see cref="Logistic"/> function at the input value.</returns>
        public static double LogisticDerivative(double X)
        {
            return X * (1 - X);
        }

        /// <summary>
        /// Suppressed negative to be 0.0, or keeps input othrwise.
        /// </summary>
        /// <param name="X">Value to suppress.</param>
        /// <returns>Suppressed value.</returns>
        public static double ReLU(double X)
        {
            if (X < 0.0)
            {
                return 0.0;
            }
            else
            {
                return X;
            }
        }

        /// <summary>
        /// Derivative of <see cref="ReLU"/> function at the particular value. Can used to find gradient.
        /// </summary>
        /// <param name="X">Value at which the derivative is to be found.</param>
        /// <returns>Derivative of the <see cref="ReLU"/> function at the input value.</returns>
        public static double ReLUDerivative(double X)
        {
            if (X < 0)
            {
                return 0;
            }
            else
            {
                return 1;
            }
        }

        /// <summary>
        /// Supresses negative input to be 0.01 portion of itself, or keeps input otherwise.
        /// </summary>
        /// <param name="X">Value to suppress</param>
        /// <returns>0.01 * X if X if less that 0, and X otherwise.</returns>
        public static double LeakyReLU(double X)
        {
            if (X < 0)
            {
                return 0.01 * X;
            }
            else
            {
                return X;
            }
        }

        /// <summary>
        /// Derivative of <see cref="LeakyReLU"/> function at the particular value. Can used to find gradient.
        /// </summary>
        /// <param name="X">Value at which the derivative is to be found.</param>
        /// <returns>Derivative of the <see cref="LeakyReLU"/> function at the input value.</returns>
        public static double LeakyReLUDerivative(double X)
        {
            if (X < 0)
            {
                return 0.01;
            }
            else
            {
                return 1;
            }
        }

        /// <summary>
        /// Returns the absolute value of a double-precision floating-point number.
        /// </summary>
        /// <param name="X">A number that is greater than or equal to System.Double.MinValue, but less than or equal to System.Double.MaxValue.</param>
        /// <returns>A double-precision floating-point number, x, such that 0 ≤ x ≤System.Double.MaxValue.</returns>
        public static double Absolute(double X)
        {
            return Math.Abs(X);
        }

        /// <summary>
        /// Resize the image to the specified width and height.
        /// </summary>
        /// <param name="image">The image to resize.</param>
        /// <param name="width">The width to resize to.</param>
        /// <param name="height">The height to resize to.</param>
        /// <returns>The resized image.</returns>
        public static Bitmap ResizeImage(Image image, int width, int height)
        {
            Rectangle destRect = new Rectangle(0, 0, width, height);
            Bitmap destImage = new Bitmap(width, height);

            destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            using (Graphics graphics = Graphics.FromImage(destImage))
            {
                graphics.CompositingMode = System.Drawing.Drawing2D.CompositingMode.SourceCopy;
                graphics.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
                graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                graphics.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
                graphics.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighQuality;

                using (System.Drawing.Imaging.ImageAttributes wrapMode = new System.Drawing.Imaging.ImageAttributes())
                {
                    wrapMode.SetWrapMode(System.Drawing.Drawing2D.WrapMode.TileFlipXY);
                    graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);
                }
            }

            return destImage;
        }

        public static int[][] GetPermutatedIndeces(params int[] dimensions)
        {
            int[][] axes = new int[dimensions.Length][];
            int totalPermutations = 1;
            for (int dimensionIndex = 0; dimensionIndex < dimensions.Length; dimensionIndex++)
            {
                totalPermutations *= dimensions[dimensionIndex];
                axes[dimensionIndex] = new int[dimensions[dimensionIndex]];
                for (int unit = 0; unit < dimensions[dimensionIndex]; unit++)
                {
                    axes[dimensionIndex][unit] = unit;
                }
            }
            int[][] permutations = new int[totalPermutations][];

            IEnumerable<IEnumerable<int>> permutationHelperFunction(IEnumerable<IEnumerable<int>> xss)
            {
                if (!xss.Any())
                {
                    return new[] { Enumerable.Empty<int>() };
                }
                else
                {
                    IEnumerable<IEnumerable<int>> query =
                        from x in xss.First()
                        from y in permutationHelperFunction(xss.Skip(1))
                        select new[] { x }.Concat(y);
                    return query;
                }
            }

            IEnumerable<int>[] enumPermutations = permutationHelperFunction(axes).ToArray();

            for (int permutationIndex = 0; permutationIndex < totalPermutations; permutationIndex++)
            {
                permutations[permutationIndex] = enumPermutations[permutationIndex].ToArray();
            }

            return permutations;
        }

        public static IList<T> TrimEndingElement<T>(IList<T> array, T element) where T : IEquatable<T>
        {
            T[] newArray = array.ToArray();

            for (int dimensionIndex = newArray.Length - 1; dimensionIndex >= 2; dimensionIndex--)
            {
                if (newArray[dimensionIndex].Equals(element))
                {
                    newArray = newArray.Take(newArray.Length - 1).ToArray();
                }
                else
                {
                    break;
                }
            }

            return newArray;
        }
    }
}
