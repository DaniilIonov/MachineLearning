using System;

namespace MachineLearning
{
    /// <summary>
    /// ThreadSafeRandom class contains methods to generate random values without cross-thread interference.
    /// </summary>
    public static class ThreadSafeRandom
    {
        private static readonly Random _global = new Random();  //Used as a global seed for a local thread Random class.

        [ThreadStatic]
        private static Random _local;   //One _local Random for each thread.

        static ThreadSafeRandom()
        {
            SetLocal(); //Calls SetLocal function before the class is used.
        }

        /// <summary>Returns a non-negative random integer.</summary>
        /// <returns>A 32-bit signed integer that is greater than or equal to 0 and less than System.Int32.MaxValue.</returns>
        public static int Next()
        {
            SetLocal();
            return _local.Next();
        }

        /// <summary>
        /// Returns a non-negative random integer that is less than the specified maximum.
        /// </summary>
        /// <param name="maxValue">The exclusive upper bound of the random number to be generated. maxValue must be greater than or equal to 0.</param>
        /// <returns>A 32-bit signed integer that is greater than or equal to 0, and less than maxValue; that is, the range of return values ordinarily includes 0 but not maxValue. However, if maxValue equals 0, maxValue is returned.</returns>
        /// <exception cref="System.ArgumentOutOfRangeException">maxValue is less than 0.</exception>
        public static int Next(int maxValue)
        {
            SetLocal();
            return _local.Next(maxValue);
        }

        /// <summary>
        /// Returns a random integer that is within a specified range.
        /// </summary>
        /// <param name="minValue">The inclusive lower bound of the random number returned.</param>
        /// <param name="maxValue">The exclusive upper bound of the random number returned. maxValue must be greater than or equal to minValue.</param>
        /// <returns>A 32-bit signed integer greater than or equal to minValue and less than maxValue; that is, the range of return values includes minValue but not maxValue. If minValue equals maxValue, minValue is returned.</returns>
        /// <exception cref="System.ArgumentOutOfRangeException">minValue is greater than maxValue.</exception>
        public static int Next(int minValue, int maxValue)
        {
            SetLocal();
            return _local.Next(minValue, maxValue);
        }

        /// <summary>
        /// Returns a random floating-point number that is greater than or equal to 0.0, and less than 1.0.
        /// </summary>
        /// <returns>A double-precision floating point number that is greater than or equal to 0.0, and less than 1.0.</returns>
        public static double NextDouble()
        {
            SetLocal();
            return _local.NextDouble();
        }

        public static double Gaussian(double mean = 0.0, double stddev = 1.0)
        {
            // The method requires sampling from a uniform random of (0,1]
            // but Random.NextDouble() returns a sample of [0,1).
            double x1 = 1 - NextDouble();
            double x2 = 1 - NextDouble();

            double y1 = Math.Sqrt(-2.0 * Math.Log(x1)) * Math.Cos(2.0 * Math.PI * x2);
            return y1 * stddev + mean;
        }

        /// <summary>
        /// SetLocal creates a local Random object for a current thread.
        /// </summary>
        private static void SetLocal()
        {
            if (_local == null) //If local does not exist in current thread.
            {
                int seed;
                lock (_global)  //Locks global Random object to create a seed for the _local object.
                {
                    seed = _global.Next();
                }
                _local = new Random(seed);  //Instantiate a _local Random object.
            }
        }
    }
}
