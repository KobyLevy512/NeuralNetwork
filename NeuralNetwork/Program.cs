

using NeuralNetwork.Core;

namespace NeuralNetwork
{
    internal class Program
    {
        static unsafe void Main(string[] args)
        {
            double[][] inputs = new double[4][]
            {
                [0,0],
                [0,1],
                [1,0],
                [1,1],
            };
            double[][] outputs = new double[4][]
            {
                [0],
                [0],
                [0],
                [1]
            };

            DeepNeural dn = new DeepNeural(1);
            dn.HiddenLayers.Add(new NeuronLayer(2, 2, 1));
            dn.ResetWeights();

            Trainer t = new Trainer();
            dn = t.Train(dn, inputs, outputs, 0.2);

            Console.WriteLine(dn.Forward(inputs[0])[0].ToString("0.00"));
            Console.WriteLine(dn.Forward(inputs[1])[0].ToString("0.00"));
            Console.WriteLine(dn.Forward(inputs[2])[0].ToString("0.00"));
            Console.WriteLine(dn.Forward(inputs[3])[0].ToString("0.00"));
        }
    }
}
