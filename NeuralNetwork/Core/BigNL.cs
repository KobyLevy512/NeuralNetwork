
using NeuralNetwork.DataModel;
using static NeuralNetwork.Core.ActivationFunction;

namespace NeuralNetwork.Core
{
    public class BigNL
    {
        private const int pinkSize = 16;
        private VirtualMatrix weightsInput;
        private VirtualMatrix weightsOutput;
        private double[] output;

        public int ExpectedOutput
        {
            get => weightsOutput.GetLength(1);
        }
        private BigNL() { }
        public BigNL(BigNL cpy)
        {
            weightsInput = new VirtualMatrix(cpy.weightsInput.GetLength(0), cpy.weightsInput.GetLength(1));
            weightsOutput = new VirtualMatrix(cpy.weightsOutput.GetLength(0), cpy.weightsOutput.GetLength(1));
            output = new double[cpy.output.Length];
        }
        public BigNL(int inputs, int nodes, int outputs)
        {
            weightsInput = new VirtualMatrix(inputs, nodes);
            weightsOutput = new VirtualMatrix(nodes, outputs);
            output = new double[nodes];
        }
        /// <summary>
        /// Reset the Weights values.
        /// </summary>
        /// <param name="rnd"></param>
        /// <param name="pink"></param>
        public void Reset(Random rnd, bool pink)
        {
            if (pink)
            {
                for (int y = 0; y < weightsInput.GetLength(0); y++)
                {
                    for (int x = 0; x < weightsInput.GetLength(1); x++)
                    {
                        double pinkSum = 0;
                        for (int i = 0; i < pinkSize; i++)
                        {
                            pinkSum += rnd.NextDouble();
                        }
                        pinkSum /= pinkSize;
                        weightsInput[y, x] = pinkSum * 2 - 1;
                    }
                }

                for (int y = 0; y < weightsOutput.GetLength(0); y++)
                {
                    for (int x = 0; x < weightsOutput.GetLength(1); x++)
                    {
                        double pinkSum = 0;
                        for (int i = 0; i < pinkSize; i++)
                        {
                            pinkSum += rnd.NextDouble();
                        }
                        pinkSum /= pinkSize;
                        weightsOutput[y, x] = pinkSum * 2 - 1;
                    }
                }
            }
            else
            {
                for (int y = 0; y < weightsInput.GetLength(0); y++)
                {
                    for (int x = 0; x < weightsInput.GetLength(1); x++)
                    {
                        weightsInput[y, x] = rnd.NextDouble() * 2 - 1;
                    }
                }
                for (int y = 0; y < weightsOutput.GetLength(0); y++)
                {
                    for (int x = 0; x < weightsOutput.GetLength(1); x++)
                    {
                        weightsOutput[y, x] = rnd.NextDouble() * 2 - 1;
                    }
                }
            }
        }

        /// <summary>
        /// Calculate this layer by inputs and pass values to the hidden layer.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="activation"></param>
        /// <exception cref="ArgumentException"></exception>
        public virtual void PassInput(double[] input, Activation activation)
        {
            if (input.Length != weightsInput.GetLength(0))
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

        /// <summary>
        /// Calculate this layer output.
        /// </summary>
        /// <param name="output"></param>
        /// <param name="activation"></param>
        /// <exception cref="ArgumentException"></exception>
        public virtual void PassOutput(double[] output, Activation activation)
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

        /// <summary>
        /// Pass inputs to the weights and get the result to the output array.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="output"></param>
        /// <param name="act"></param>
        public void Pass(double[] input, double[] output, Activation act)
        {
            PassInput(input, act);
            PassOutput(output, act);
        }

        /// <summary>
        /// Update the weights in Backpropagate.
        /// </summary>
        /// <param name="errorDelats"></param>
        public virtual void UpdateWeights(double[] inputs, double[] outputDeltas, Activation actDer, double learningRate)
        {
            //Calculate deltas.
            double[] hiddenDeltas = new double[output.Length];
            for (int i = 0; i < output.Length; i++)
            {
                double error = 0.0;
                for (int j = 0; j < outputDeltas.Length; j++)
                {
                    error += outputDeltas[j] * weightsOutput[i, j];
                }
                hiddenDeltas[i] = error * actDer(output[i]);
            }

            //Update output weights.
            for (int i = 0; i < weightsOutput.GetLength(0); i++)
            {
                for (int j = 0; j < weightsOutput.GetLength(1); j++)
                {
                    weightsOutput[i, j] += learningRate * outputDeltas[j] * output[i];
                }
            }

            //Update input weights.
            for (int i = 0; i < weightsInput.GetLength(0); i++)
            {
                for (int j = 0; j < weightsInput.GetLength(1); j++)
                {
                    weightsInput[i, j] += learningRate * hiddenDeltas[j] * inputs[i];
                }
            }
        }

        public void SaveToFile(BinaryWriter bw)
        {
            bw.Write(output.Length);
            for (int i = 0; i < output.Length; i++)
            {
                bw.Write(output[i]);
            }

            bw.Write(weightsOutput.GetLength(0));
            bw.Write(weightsOutput.GetLength(1));
            for (int y = 0; y < weightsOutput.GetLength(0); y++)
            {
                for (int x = 0; x < weightsOutput.GetLength(1); x++)
                {
                    bw.Write(weightsOutput[y, x]);
                }
            }

            bw.Write(weightsInput.GetLength(0));
            bw.Write(weightsInput.GetLength(1));
            for (int y = 0; y < weightsInput.GetLength(0); y++)
            {
                for (int x = 0; x < weightsInput.GetLength(1); x++)
                {
                    bw.Write(weightsInput[y, x]);
                }
            }
        }

        public static BigNL LoadFromFile(BinaryReader br)
        {
            BigNL ret = new BigNL();

            ret.output = new double[br.ReadInt32()];
            for (int i = 0; i < ret.output.Length; i++)
            {
                ret.output[i] = br.ReadDouble();
            }

            ret.weightsOutput = new VirtualMatrix(br.ReadInt32(), br.ReadInt32());
            for (int y = 0; y < ret.weightsOutput.GetLength(0); y++)
            {
                for (int x = 0; x < ret.weightsOutput.GetLength(1); x++)
                {
                    ret.weightsOutput[y, x] = br.ReadDouble();
                }
            }

            ret.weightsInput = new VirtualMatrix(br.ReadInt32(), br.ReadInt32());
            for (int y = 0; y < ret.weightsInput.GetLength(0); y++)
            {
                for (int x = 0; x < ret.weightsInput.GetLength(1); x++)
                {
                    ret.weightsInput[y, x] = br.ReadDouble();
                }
            }
            return ret;
        }
    }
}
