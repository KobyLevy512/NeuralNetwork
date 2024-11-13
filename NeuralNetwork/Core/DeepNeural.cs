
using NeuralNetwork.DataModel;
using static NeuralNetwork.Core.ActivationFunction;

namespace NeuralNetwork.Core
{
    public class DeepNeural
    {
        // Random generator for weight initialization
        private Random random;

        //Layers
        public List<NeuronLayer> HiddenLayers;
        private double[] output;
        private double learningRate;

        //Learning Info.
        public Activation Activation, ActivationDerivative;
        public int Seed;

        private DeepNeural()
        {
            HiddenLayers = new List<NeuronLayer>();
        }
        public DeepNeural(DeepNeural cpy, int newSeed = 0, double learningRate = 0)
        {
            //Set the new seed if supply
            if (newSeed != 0) Seed = newSeed;
            else Seed = cpy.Seed;
            random = new Random(Seed);

            //Set the learning rate.
            if(learningRate != 0)
                this.learningRate = learningRate;
            else this.learningRate = cpy.learningRate;

            //Set Activations.
            Activation = cpy.Activation;
            ActivationDerivative = cpy.ActivationDerivative;

            //Set the layers.
            output = new double[cpy.output.Length];
            HiddenLayers = new List<NeuronLayer>();
            for (int i = 0; i < cpy.HiddenLayers.Count; i++)
            {
                HiddenLayers.Add(new NeuronLayer(cpy.HiddenLayers[i]));
            }

            ResetWeights();
            
        }
        public DeepNeural(int outputSize):this(outputSize, 0) { }
        public DeepNeural(int outputSize, int seed):this(outputSize, seed, Sigmoid, SigmoidDerivative) { }
        public DeepNeural(int outputsSize, int seed, Activation act, Activation der)
        {
            //Set the layers.
            HiddenLayers = new List<NeuronLayer>();
            output = new double[outputsSize];

            //Set activations.
            Activation = act;
            ActivationDerivative = der;

            //Apply the seed.
            Seed = seed;
            if(Seed != 0)
            {
                random = new Random(Seed);
            }
            else
            {
                random = new Random();
            }

            this.learningRate = 0.1;
        }

        /// <summary>
        /// Reset all the weights.
        /// </summary>
        /// <param name="pink"></param>
        public void ResetWeights(bool pink = false)
        {
            for(int i = 0; i < HiddenLayers.Count; i++) 
            {
                HiddenLayers[i].Reset(random, pink);
            }
        }

        public double[] Forward(double[] inputs)
        {   
            //Pass all the hidden layers.
            for(int i = 0; i < HiddenLayers.Count; i++)
            {
                double[] output = new double[HiddenLayers[i].ExpectedOutput];
                HiddenLayers[i].Pass(inputs, output, Activation);
                inputs = output;
            }

            //Apply the result and return.
            output = inputs;
            return output;
        }

        public void Train(double[] inputs, double[] targets, int epochs)
        {
            for(int epoch = 0; epoch < epochs; epoch++)
            {
                //Pass all the hidden layers and update them.
                for (int i = 0; i < HiddenLayers.Count; i++)
                {
                    double[] output = new double[HiddenLayers[i].ExpectedOutput];
                    HiddenLayers[i].Pass(inputs, output, Activation);

                    //Calculate output deltas.
                    double[] outputDeltas = new double[output.Length];
                    for (int j = 0; j < outputDeltas.Length; j++)
                    {
                        double error = targets[j] - output[j];
                        outputDeltas[j] = error * ActivationDerivative(output[j]);
                    }

                    HiddenLayers[i].UpdateWeights(inputs, outputDeltas, ActivationDerivative, learningRate);
                    inputs = output;
                }
            }
        }

        public void Train(double[][] inputs, double[][] targets, int epochs)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                for(int i = 0; i < inputs.Length; i++)
                {
                    Train(inputs[i], targets[i], 1);
                }
            }
        }

        public void Train(DataStream stream, int epochs)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                for (int i = 0; i < stream.Count; i++)
                {
                    var entry = stream.ReadEntry(i);
                    Train(entry.Item1, entry.Item2, 1);
                }
            }
        }
        public virtual double Test(double[] inputs, double[] targets)
        {
            double error = 0;
            double[] output = Forward(inputs);
            for (int i = 0; i < output.Length; i++)
            {
                error += Math.Abs(targets[i] - output[i]);
            }
            return error;
        }

        public double Test(double[][] inputs, double[][] targets)
        {
            double error = 0;
            for(int i = 0; i <inputs.Length; i++)
            {
                error += Test(inputs[i], targets[i]);
            }
            return error;
        }

        public double Test(DataStream stream)
        {
            double error = 0;
            for (int i = 0; i < stream.Count; i++)
            {
                var entry = stream.ReadEntry(i);
                error += Test(entry.Item1, entry.Item2);
            }
            return error;
        }
        public void Print(double[][] inputs)
        {
            for (int i = 0; i < inputs.GetLength(0); i++)
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

        public void SaveToFile(string filename)
        {
            BinaryWriter bw = new BinaryWriter(File.Create(filename));

            //Train attributes.
            bw.Write(Seed);
            bw.Write(learningRate);

            //Writing the activation functions.
            for (int i = 0; i < Functions.Length; i++)
            {
                if (Activation == Functions[i])
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

            //Output
            bw.Write(output.Length);
            for (int i = 0; i < output.Length; i++)
            {
                bw.Write(output[i]);
            }

            //Layers
            bw.Write(HiddenLayers.Count);
            for(int i = 0; i < HiddenLayers.Count; i++)
            {
                HiddenLayers[i].SaveToFile(bw);
            }

            bw.Close();
        }

        public static DeepNeural LoadFromFile(string filename)
        {
            DeepNeural ret = new DeepNeural();
            BinaryReader br = new BinaryReader(File.OpenRead(filename));

            //Train attributes.
            ret.Seed = br.ReadInt32();
            ret.learningRate = br.ReadDouble();

            //Activation Functions.
            ret.Activation = Functions[br.ReadInt32()];
            ret.ActivationDerivative = Functions[br.ReadInt32()];

            //Output.
            ret.output = new double[br.ReadInt32()];
            for (int i = 0; i < ret.output.Length; i++)
            {
                ret.output[i] = br.ReadDouble();
            }

            //Layers.
            int count = br.ReadInt32();
            for(int i = 0; i < count; i++)
            {
                ret.HiddenLayers.Add(NeuronLayer.LoadFromFile(br));
            }

            br.Close();
            return ret;
        }
    }
}
