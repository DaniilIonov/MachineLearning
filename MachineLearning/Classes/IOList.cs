using System;
using System.Collections.Generic;

namespace MachineLearning
{
    /// <summary>
    /// Represents a collection of input-output pairs that can be accessed by index.
    /// </summary>
    [Serializable]
    public class IOList : System.Collections.ObjectModel.Collection<Dictionary<String, List<double>>>
    {
        /// <summary>
        /// Gets the number of elements actually contained in the <see cref="IOList"/>
        /// </summary>
        public new int Count
        {
            get { return base.Count; }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="IOList"/> class.
        /// </summary>
        public IOList() : base()
        {

        }

        /// <summary>
        /// Initializes a new instance of the <see cref="IOList"/> class as a wrapper for the specified list.
        /// </summary>
        /// <param name="list">The list that is wrapped by the new collection.</param>
        /// <exception cref="ArgumentNullException">list is null.</exception>
        public IOList(IList<Dictionary<String, List<double>>> list) : base(list)
        {

        }

        /// <summary>
        /// Adds an input-output pair to the end of the <see cref="IOList"/>.
        /// </summary>
        /// <param name="_Inputs">The input set to be added to the end of the <see cref="IOList"/>.</param>
        /// <param name="_Outputs">The output set to be added to the end of the <see cref="IOList"/>.</param>
        public void Add(List<double> _Inputs, List<double> _Outputs)
        {
            Dictionary<String, List<double>> dictionary = new Dictionary<string, List<double>>();
            dictionary.Add("Inputs", _Inputs);
            dictionary.Add("Outputs", _Outputs);
            base.Add(dictionary);
        }

        /// <summary>
        /// Returns a string containing the current <see cref="IOList.Count"/>.
        /// </summary>
        /// <returns>A string in format "Count = <see cref="IOList.Count"/>.</returns>
        public override string ToString()
        {
            return String.Format("Count = {0}", Count);
        }

        /// <summary>
        /// Shuffles the <see cref="IOList"/>
        /// </summary>
        public void Shuffle()
        {
            var n = base.Count;
            while (n > 1)
            {
                n--;
                var k = ThreadSafeRandom.Next(n + 1);
                var value = base[k];
                base[k] = base[n];
                base[n] = value;
            }
        }
    }
}
