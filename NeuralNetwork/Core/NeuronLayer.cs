
using static NeuralNetwork.Core.ActivationFunction;

namespace NeuralNetwork.Core
{
    public class NeuronLayer
    {
        private double[,] weightsInput;
        private double[,] weightsOutput;
        private double[] output;

        public NeuronLayer(NeuronLayer cpy)
        {
            weightsInput = new double[cpy.weightsInput.GetLength(0), cpy.weightsInput.GetLength(1)];
            weightsOutput = new double[cpy.weightsOutput.GetLength(0), cpy.weightsOutput.GetLength(1)];
            output = new double[cpy.output.Length];
        }
        public NeuronLayer(int inputs, int nodes, int outputs)
        {
            weightsInput = new double[inputs, nodes];
            weightsOutput = new double[nodes, outputs];
            output = new double[nodes];   
        }

        public void PassInput(double[] input, Activation activation)
        {
            if(input.Length != weightsInput.GetLength(0))
            {
                throw new ArgumentException("inputs array size is not match");
            }

            for (int i = 0; i < output.Length; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < input.Length; j++)
                {
                    sum += input[j] * weightsInput[j, i];
                }
                output[i] = activation.Invoke(sum);
            }
        }

        public void PassOutput(double[] output, Activation activation)
        {
            if (output.Length != weightsOutput.GetLength(1))
            {
                throw new ArgumentException("inputs array size is not match");
            }
            for (int i = 0; i < output.Length; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < this.output.Length; j++)
                {
                    sum += this.output[j] * weightsOutput[j, i];
                }
                output[i] = activation.Invoke(sum);
            }
        }
    }
}
