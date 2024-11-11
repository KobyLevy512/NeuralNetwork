
using NeuralNetwork.DataModel;
using static NeuralNetwork.Core.ActivationFunction;

namespace NeuralNetwork.Core
{
    public class NeuralNetwork
    {
        // Random generator for weight initialization
        private Random random = new Random();

        // Network parameters
        private double[,] weightsInputHidden;
        private double[,] weightsHiddenOutput;
        private double[] hiddenLayerOutput;
        private double[] outputLayerOutput;
        private double learningRate;

        public ActivationFunction.Activation Activation, ActivationDerivative;
        public int Seed;
        private NeuralNetwork(){}
        public NeuralNetwork(DataModelBase model) : this(model.GetInputsSize(), model.GetInputsSize(), model.GetTargetsSize())
        {

        }
        public NeuralNetwork(NeuralNetwork cpy, int seed = 0, double learningRate = 0.0)
        {
            if(learningRate == 0.0)
                this.learningRate = cpy.learningRate;
            else 
                this.learningRate = learningRate;

            if(seed != 0)
                random = new Random(seed);

            this.Activation = cpy.Activation;
            this.ActivationDerivative = cpy.ActivationDerivative;

            // Initialize weights with small random values
            weightsInputHidden = new double[cpy.weightsInputHidden.GetLength(0), cpy.weightsInputHidden.GetLength(1)];
            weightsHiddenOutput = new double[cpy.weightsHiddenOutput.GetLength(0), cpy.weightsHiddenOutput.GetLength(1)];
            hiddenLayerOutput = new double[cpy.hiddenLayerOutput.Length];
            outputLayerOutput = new double[cpy.outputLayerOutput.Length];
            Seed = seed;

            InitializeWeights(weightsInputHidden);
            InitializeWeights(weightsHiddenOutput);
        }
        public NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, int seed = 0, double learningRate = 0.1)
        {
            if(seed != 0)
            {
                random = new Random(seed);
            }
            Activation = ActivationFunction.Sigmoid;
            ActivationDerivative = ActivationFunction.SigmoidDerivative;

            this.learningRate = learningRate;

            // Initialize weights with small random values
            weightsInputHidden = new double[inputNodes, hiddenNodes];
            weightsHiddenOutput = new double[hiddenNodes, outputNodes];
            hiddenLayerOutput = new double[hiddenNodes];
            outputLayerOutput = new double[outputNodes];
            Seed = seed;

            InitializeWeights(weightsInputHidden);
            InitializeWeights(weightsHiddenOutput);
        }

        private void InitializeWeights(double[,] weights)
        {
            for (int i = 0; i < weights.GetLength(0); i++)
            {
                for (int j = 0; j < weights.GetLength(1); j++)
                {
                    double noise = 0;
                    for(int i = 0; i < 16; i++)
                    {
                        noise += random.NextDouble();
                    }
                    noise /= 16;
                    weights[i, j] = noise * 2 - 1; // Range [-1, 1]
                }
            }
        }

        public double[] Forward(double[] inputs)
        {
            // Input -> Hidden
            for (int i = 0; i < hiddenLayerOutput.Length; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < inputs.Length; j++)
                {
                    sum += inputs[j] * weightsInputHidden[j, i];
                }
                hiddenLayerOutput[i] = Activation.Invoke(sum);
            }

            // Hidden -> Output
            for (int i = 0; i < outputLayerOutput.Length; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < hiddenLayerOutput.Length; j++)
                {
                    sum += hiddenLayerOutput[j] * weightsHiddenOutput[j, i];
                }
                outputLayerOutput[i] = Activation.Invoke(sum);
            }

            return outputLayerOutput;
        }

        public double[] Forward(DataModelBase model)
        {
            return Forward(model.GetInput());
        }

        public void Backpropagate(double[] inputs, double[] targets)
        {
            // Calculate output layer error and deltas
            double[] outputDeltas = new double[outputLayerOutput.Length];
            for (int i = 0; i < outputDeltas.Length; i++)
            {
                double error = targets[i] - outputLayerOutput[i];
                outputDeltas[i] = error * ActivationDerivative(outputLayerOutput[i]);
            }

            // Calculate hidden layer error and deltas
            double[] hiddenDeltas = new double[hiddenLayerOutput.Length];
            for (int i = 0; i < hiddenLayerOutput.Length; i++)
            {
                double error = 0.0;
                for (int j = 0; j < outputDeltas.Length; j++)
                {
                    error += outputDeltas[j] * weightsHiddenOutput[i, j];
                }
                hiddenDeltas[i] = error * ActivationDerivative(hiddenLayerOutput[i]);
            }

            // Update weights hidden -> output
            for (int i = 0; i < weightsHiddenOutput.GetLength(0); i++)
            {
                for (int j = 0; j < weightsHiddenOutput.GetLength(1); j++)
                {
                    weightsHiddenOutput[i, j] += learningRate * outputDeltas[j] * hiddenLayerOutput[i];
                }
            }

            // Update weights input -> hidden
            for (int i = 0; i < weightsInputHidden.GetLength(0); i++)
            {
                for (int j = 0; j < weightsInputHidden.GetLength(1); j++)
                {
                    weightsInputHidden[i, j] += learningRate * hiddenDeltas[j] * inputs[i];
                }
            }
        }

        public void Backpropagate(DataModelBase model)
        {
            Backpropagate(model.GetInput(), model.GetTarget());
        }

        public void Train(double[] inputs, double[] targets, int epochs)
        {
            for (int i = 0; i < epochs; i++)
            {
                Forward(inputs);
                Backpropagate(inputs, targets);
            }
        }
        public void Train(double[][] inputs, double[][] targets, int epochs)
        {
            for(int j = 0; j < epochs; j++)
            {
                for (int i = 0; i < inputs.Length; i++)
                {
                    Forward(inputs[i]);
                    Backpropagate(inputs[i], targets[i]);
                }
            }
        }

        public void Train(DataModelBase model, int epochs)
        {
            Train(model.GetInput(), model.GetTarget(), epochs);
        }

        public void Train<T>(T[] model, int epochs) where T : DataModelBase
        {
            for(int i = 0; i < model.Length; i++)
            {
                Train(model[i], epochs);
            }
        }

        public double Test(double[] inputs, double[] targets)
        {
            double error = 0;
            double[] output = Forward(inputs);
            for (int i = 0; i < output.Length; i++)
            {
                error += Math.Abs(targets[i] - output[i]);
            }
            return error;
        }
        public double Test(DataModelBase model)
        {
            return Test(model.GetInput(), model.GetTarget());
        }

        public double Test(double[][] inputs, double[][] targets)
        {
            double error = 0;
            for(int i = 0; i < inputs.Length; i++)
            {
                error += Test(inputs[i], targets[i]);
            }
            return error;
        }

        public double Test<T>(T[] model) where T : DataModelBase
        {
            double error = 0;
            for (int i = 0; i < model.Length; i++)
            {
                error += Test(model[i]);
            }
            return error;
        }
        public void Print(double[][] inputs)
        {
            for(int i = 0;i < inputs.GetLength(0);i++)
            {
                double[] res = Forward(inputs[i]);
                Console.Write("{");
                int j = 0;
                for (; j < inputs[i].Length - 1; j++)
                {
                    Console.Write(inputs[i][j].ToString("0.00") + ", ");
                }
                Console.Write(inputs[i][j].ToString("0.00"));
                Console.Write("} >> {");
                for (j = 0; j < res.Length - 1; j++)
                {
                    Console.Write(res[j].ToString("0.00") + ", ");
                }
                Console.WriteLine(res[j].ToString("0.00") + "}");
            }
        }

        public void SaveToFile(string path)
        {
            BinaryWriter bw = new BinaryWriter(File.Create(path));

            bw.Write(Seed);

            for(int i = 0; i < Functions.Length; i++) 
            {
                if(Activation == Functions[i])
                {
                    bw.Write(i);
                    break;
                }
            }

            for (int i = 0; i < Functions.Length; i++)
            {
                if (ActivationDerivative == Functions[i])
                {
                    bw.Write(i);
                    break;
                }
            }

            bw.Write(learningRate);

            bw.Write(outputLayerOutput.Length);
            foreach(double v in outputLayerOutput)
            {
                bw.Write(v);
            }

            bw.Write(hiddenLayerOutput.Length);
            foreach (double v in hiddenLayerOutput)
            {
                bw.Write(v);
            }

            bw.Write(weightsHiddenOutput.GetLength(0));
            bw.Write(weightsHiddenOutput.GetLength(1));
            for(int y = 0; y < weightsHiddenOutput.GetLength(0); y++)
            {
                for(int x = 0; x < weightsHiddenOutput.GetLength(1); x++)
                {
                    bw.Write(weightsHiddenOutput[y, x]);
                }
            }

            bw.Write(weightsInputHidden.GetLength(0));
            bw.Write(weightsInputHidden.GetLength(1));
            for (int y = 0; y < weightsInputHidden.GetLength(0); y++)
            {
                for (int x = 0; x < weightsInputHidden.GetLength(1); x++)
                {
                    bw.Write(weightsInputHidden[y, x]);
                }
            }
            bw.Close();
        }

        public static NeuralNetwork LoadFile(string filename)
        {
            BinaryReader br = new BinaryReader(File.Open(filename, FileMode.Open));
            NeuralNetwork nw = new NeuralNetwork();
            nw.Seed = br.ReadInt32();
            nw.Activation = Functions[br.ReadInt32()];
            nw.ActivationDerivative = Functions[br.ReadInt32()];
            nw.learningRate = br.ReadDouble();
            nw.outputLayerOutput = new double[br.ReadInt32()];
            for(int i = 0; i < nw.outputLayerOutput.Length; i++)
            {
                nw.outputLayerOutput[i] = br.ReadDouble();
            }
            nw.hiddenLayerOutput = new double[br.ReadInt32()];
            for (int i = 0; i < nw.hiddenLayerOutput.Length; i++)
            {
                nw.hiddenLayerOutput[i] = br.ReadDouble();
            }
            nw.weightsHiddenOutput = new double[br.ReadInt32(), br.ReadInt32()];
            for(int y = 0; y < nw.weightsHiddenOutput.GetLength(0); y++)
            {
                for(int x = 0; x < nw.weightsHiddenOutput.GetLength(1); x++)
                {
                    nw.weightsHiddenOutput[y,x] = br.ReadDouble();
                }
            }
            nw.weightsInputHidden = new double[br.ReadInt32(), br.ReadInt32()];
            for (int y = 0; y < nw.weightsInputHidden.GetLength(0); y++)
            {
                for (int x = 0; x < nw.weightsInputHidden.GetLength(1); x++)
                {
                    nw.weightsInputHidden[y, x] = br.ReadDouble();
                }
            }
            br.Close();
            return nw;
        }

        private class inputs
        {
        }
    }
}
