

using NeuralNetwork.Core;
using NeuralNetwork.DataModel;
using NeuralNetwork.Utils;

namespace NeuralNetwork
{
    internal class Program
    {
        static unsafe void Main(string[] args)
        {
            QuestionAnswerModel model = new QuestionAnswerModel("What is your age", "my age is 29");
            Core.NeuralNetwork nn = new Core.NeuralNetwork(model);
            nn.Train(model, 10000);
            Console.WriteLine(Parser.DoubleToString(nn.Forward(model)));
        }
    }
}
