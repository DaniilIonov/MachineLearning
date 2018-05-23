using System;
using System.Collections.Generic;

namespace MachineLearning
{
    /// <summary>
    /// Represents a collection of input-output pairs that can be accessed by index.
    /// </summary>
    [Serializable]
    public class IOList : System.Collections.ObjectModel.Collection<Dictionary<string, Matrix>>
    {
        /// <summary>
        /// Gets the number of elements actually contained in the <see cref="IOList"/>.
        /// </summary>
        public new int Count
        {
            get
            {
                return base.Count;
            }
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
        public IOList(IList<Dictionary<string, Matrix>> list) : base(list)
        {

        }

        /// <summary>
        /// Adds an input-output pair to the end of the <see cref="IOList"/>.
        /// </summary>
        /// <param name="inputs">The input set to be added to the end of the <see cref="IOList"/>.</param>
        /// <param name="outputs">The output set to be added to the end of the <see cref="IOList"/>.</param>
        public void Add(Matrix inputs, Matrix outputs)
        {
            Dictionary<string, Matrix> dictionary = new Dictionary<string, Matrix>
            {
                { "Inputs", inputs },
                { "Outputs", outputs }
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
        /// Shuffles the <see cref="IOList"/>.
        /// </summary>
        public void Shuffle()
        {
            int n = this.Count;
            while (n > 1)
            {
                n--;
                int k = ThreadSafeRandom.Next(n + 1);
                Dictionary<string, Matrix> value = this[k];
                this[k] = this[n];
                this[n] = value;
            }
        }
    }
}
