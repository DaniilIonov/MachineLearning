using System;
using System.Collections.Generic;

namespace MachineLearning
{
    /// <summary>
    /// Represents a collection of input-output pairs that can be accessed by index.
    /// </summary>
    [Serializable]
    public class IOList : System.Collections.ObjectModel.Collection<Dictionary<string, List<double>>>
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
        public IOList(IList<Dictionary<string, List<double>>> list) : base(list)
        {

        }

        /// <summary>
        /// Adds an input-output pair to the end of the <see cref="IOList"/>.
        /// </summary>
        /// <param name="_Inputs">The input set to be added to the end of the <see cref="IOList"/>.</param>
        /// <param name="_Outputs">The output set to be added to the end of the <see cref="IOList"/>.</param>
        public void Add(List<double> _Inputs, List<double> _Outputs)
        {
            Dictionary<string, List<double>> dictionary = new Dictionary<string, List<double>>
            {
                { "Inputs", _Inputs },
                { "Outputs", _Outputs }
            };
            base.Add(dictionary);
        }

        /// <summary>
        /// Returns a string containing the current <see cref="IOList.Count"/>.
        /// </summary>
        /// <returns>A string in format "Count = <see cref="IOList.Count"/>.</returns>
        public override string ToString()
        {
            return string.Format("Count = {0}", this.Count);
        }

        /// <summary>
        /// Shuffles the <see cref="IOList"/>
        /// </summary>
        public void Shuffle()
        {
            int n = base.Count;
            while (n > 1)
            {
                n--;
                int k = ThreadSafeRandom.Next(n + 1);
                Dictionary<string, List<double>> value = base[k];
                base[k] = base[n];
                base[n] = value;
            }
        }
    }
}
