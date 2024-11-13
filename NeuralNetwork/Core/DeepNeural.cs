
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
    }
}
