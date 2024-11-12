
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

        //Learning Info.
        public Activation Activation, ActivationDerivative;
        public int Seed;

        public DeepNeural(DeepNeural cpy, int newSeed = 0)
        {
            if (newSeed != 0) Seed = newSeed;
            else Seed = cpy.Seed;
            random = new Random(Seed);

            Activation = cpy.Activation;
            ActivationDerivative = cpy.ActivationDerivative;

            output = new double[cpy.output.Length];

            HiddenLayers = new List<NeuronLayer>();
            for(int i = 0; i < cpy.HiddenLayers.Count; i++)
            {
                HiddenLayers.Add(new NeuronLayer(cpy.HiddenLayers[i]));
            }
        }
        public DeepNeural(int outputSize):this(outputSize, 0) { }
        public DeepNeural(int outputSize, int seed):this(outputSize, seed, Sigmoid, SigmoidDerivative) { }
        public DeepNeural(int outputsSize, int seed, Activation act, Activation der)
        {
            HiddenLayers = new List<NeuronLayer>();
            output = new double[outputsSize];
            Activation = act;
            ActivationDerivative = der;

            if(seed != 0)
            {
                random = new Random(seed);
            }
            else
            {
                random = new Random();
            }
        }
    }
}
