namespace MachineLearning
{
    /// <summary>
    /// Defines a Neural Layer Interface to be used in <see cref="NeuralNetwork"/> class.
    /// </summary>
    public interface INeuralLayer
    {
        int NumberOfInputs { get; }

        int NumberOfOutputs { get; }

        double[] FeedForward(double[] _Inputs);

        void BackPropagation(double[] Error);

        void BackPropagation(double[] PropagatedErrorOuter, double[,] WeightsOuter);

        void RandomizeWeights();

        void CorrectWeights(double LearningRate, double regularizationRate);
    }
}
