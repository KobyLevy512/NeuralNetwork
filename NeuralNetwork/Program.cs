

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
            Core.NeuralNetwork nw = new Core.NeuralNetwork(2, 2, 1, -944892808);
            nw.Activation = ActivationFunction.ReLU;
            nw.ActivationDerivative = ActivationFunction.ReLUDerivative;

            DeepNeural dn = new DeepNeural(1, -944892808);
            dn.Activation = ActivationFunction.ReLU;
            dn.ActivationDerivative = ActivationFunction.ReLUDerivative;
            dn.HiddenLayers.Add(new NeuronLayer(2, 2, 1));
            dn.ResetWeights();

            nw.Train(inputs, outputs, 10000);
            dn.Train(inputs, outputs, 10000, 0.1);

            Console.WriteLine(nw.Forward(inputs[0])[0]);
            Console.WriteLine(nw.Forward(inputs[1])[0]);
            Console.WriteLine(nw.Forward(inputs[2])[0]);
            Console.WriteLine(nw.Forward(inputs[3])[0]);

            Console.WriteLine(dn.Forward(inputs[0])[0].ToString("0.00"));
            Console.WriteLine(dn.Forward(inputs[1])[0].ToString("0.00"));
            Console.WriteLine(dn.Forward(inputs[2])[0].ToString("0.00"));
            Console.WriteLine(dn.Forward(inputs[3])[0].ToString("0.00"));
        }
    }
}
