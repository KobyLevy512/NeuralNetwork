using NeuralNetwork.Core;
using System.Diagnostics;
using System.Numerics;

namespace NeuralNetwork
{
    internal class Program
    {
        static unsafe void Main(string[] args)
        {
            int cur = 64;
            double* curPtr = (double*)&cur;
            double value = *curPtr;
            Console.WriteLine(value);
        }
    }
}
