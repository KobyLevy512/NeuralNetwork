
using NeuralNetwork.DataModel;
using static NeuralNetwork.Core.ActivationFunction;

namespace NeuralNetwork.Core
{
    public unsafe class BigNL
    {
        private const int pinkSize = 16;
        private VirtualMatrix weightsInput;
        private VirtualMatrix weightsOutput;
        private double[] output;

        public int ExpectedOutput
        {
            get => weightsOutput.GetLength1;
        }
        private BigNL() { }
        public BigNL(BigNL cpy)
        {
            weightsInput = new VirtualMatrix(cpy.weightsInput.GetLength0, cpy.weightsInput.GetLength1);
            weightsOutput = new VirtualMatrix(cpy.weightsOutput.GetLength0, cpy.weightsOutput.GetLength1);
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
                for (int y = 0; y < weightsInput.GetLength0; y++)
                {
                    double* row = weightsInput.ReadRow(y);
                    for (int x = 0; x < weightsInput.GetLength1; x++)
                    {
                        double pinkSum = 0;
                        for (int i = 0; i < pinkSize; i++)
                        {
                            pinkSum += rnd.NextDouble();
                        }
                        pinkSum /= pinkSize;
                        *row = pinkSum * 2 - 1;
                        row++;
                    }
                }

                for (int y = 0; y < weightsOutput.GetLength0; y++)
                {
                    double* row = weightsOutput.ReadRow(y);
                    for (int x = 0; x < weightsOutput.GetLength1; x++)
                    {
                        double pinkSum = 0;
                        for (int i = 0; i < pinkSize; i++)
                        {
                            pinkSum += rnd.NextDouble();
                        }
                        pinkSum /= pinkSize;
                        *row = pinkSum * 2 - 1;
                        row++;
                    }
                }
            }
            else
            {
                for (int y = 0; y < weightsInput.GetLength0; y++)
                {
                    double* row = weightsInput.ReadRow(y);
                    for (int x = 0; x < weightsInput.GetLength1; x++)
                    {
                        *row = rnd.NextDouble() * 2 - 1;
                        row++;
                    }
                }

                for (int y = 0; y < weightsOutput.GetLength0; y++)
                {
                    double* row = weightsOutput.ReadRow(y);
                    for (int x = 0; x < weightsOutput.GetLength1; x++)
                    {
                        *row = rnd.NextDouble() * 2 - 1;
                        row++;
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
            if (input.Length != weightsInput.GetLength0)
            {
                throw new ArgumentException("inputs array size is not match");
            }

            for(int i = 0; i < output.Length; i++)
            {
                output[i] = 0;
            }

            for (int j = 0; j < input.Length; j++)
            {
                double* row = weightsInput.ReadRow(j);
                for(int i = 0; i < output.Length; i++)
                {
                    output[i] += input[j] * row[i];
                }
            }

            for (int i = 0; i < output.Length; i++)
            {
                output[i] = activation.Invoke(output[i]);
            }

            //for (int i = 0; i < output.Length; i++)
            //{
            //    double sum = 0.0;
            //    for (int j = 0; j < input.Length; j++)
            //    {
            //        sum += input[j] * weightsInput[j, i];
            //    }
            //    output[i] = activation.Invoke(sum);
            //}
        }

        /// <summary>
        /// Calculate this layer output.
        /// </summary>
        /// <param name="output"></param>
        /// <param name="activation"></param>
        /// <exception cref="ArgumentException"></exception>
        public virtual void PassOutput(double[] output, Activation activation)
        {
            if (output.Length != weightsOutput.GetLength1)
            {
                throw new ArgumentException("inputs array size is not match");
            }

            for (int i = 0; i < output.Length; i++)
            {
                output[i] = 0;
            }

            for (int j = 0; j < this.output.Length; j++)
            {
                double* row = weightsInput.ReadRow(j);
                for (int i = 0; i < output.Length; i++)
                {
                    output[i] += this.output[j] * row[i];
                }
            }

            for (int i = 0; i < output.Length; i++)
            {
                output[i] = activation.Invoke(output[i]);
            }

            //for (int i = 0; i < output.Length; i++)
            //{
            //    double sum = 0.0;
            //    for (int j = 0; j < this.output.Length; j++)
            //    {
            //        sum += this.output[j] * weightsOutput[j, i];
            //    }
            //    output[i] = activation.Invoke(sum);
            //}
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
                double* row = weightsOutput.ReadRow(i);
                for (int j = 0; j < outputDeltas.Length; j++)
                {
                    error += outputDeltas[j] * row[j];
                }
                hiddenDeltas[i] = error * actDer(output[i]);
            }

            //Update output weights.
            for (int i = 0; i < weightsOutput.GetLength0; i++)
            {
                double* row = weightsOutput.ReadRow(i);
                for (int j = 0; j < weightsOutput.GetLength1; j++)
                {
                    row[j] += learningRate * outputDeltas[j] * output[i];
                }
                weightsOutput.WriteRow(i);
            }

            //Update input weights.
            for (int i = 0; i < weightsInput.GetLength0; i++)
            {
                double* row = weightsInput.ReadRow(i);
                for (int j = 0; j < weightsInput.GetLength1; j++)
                {
                    row[j] += learningRate * hiddenDeltas[j] * inputs[i];
                }
                weightsInput.WriteRow(i);
            }
        }

        public void SaveToFile(BinaryWriter bw)
        {
            bw.Write(output.Length);
            for (int i = 0; i < output.Length; i++)
            {
                bw.Write(output[i]);
            }

            bw.Write(weightsOutput.GetLength0);
            bw.Write(weightsOutput.GetLength1);
            for (int y = 0; y < weightsOutput.GetLength0; y++)
            {
                double* row = weightsOutput.ReadRow(y);
                for (int x = 0; x < weightsOutput.GetLength1; x++)
                {
                    bw.Write(row[x]);
                }
            }

            bw.Write(weightsInput.GetLength0);
            bw.Write(weightsInput.GetLength0);
            for (int y = 0; y < weightsInput.GetLength0; y++)
            {
                double* row = weightsInput.ReadRow(y);
                for (int x = 0; x < weightsInput.GetLength1; x++)
                {
                    bw.Write(row[x]);
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
            for (int y = 0; y < ret.weightsOutput.GetLength0; y++)
            {
                double* row = ret.weightsOutput.ReadRow(y);
                for (int x = 0; x < ret.weightsOutput.GetLength1; x++)
                {
                    row[x] = br.ReadDouble();
                }
                ret.weightsOutput.WriteRow(y);
            }

            ret.weightsInput = new VirtualMatrix(br.ReadInt32(), br.ReadInt32());
            for (int y = 0; y < ret.weightsInput.GetLength0; y++)
            {
                double* row = ret.weightsInput.ReadRow(y);
                for (int x = 0; x < ret.weightsInput.GetLength1; x++)
                {
                    row[x] = br.ReadDouble();
                }
                ret.weightsInput.WriteRow(y);
            }
            return ret;
        }
    }
}
