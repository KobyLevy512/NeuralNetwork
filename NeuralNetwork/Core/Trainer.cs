
using System.Diagnostics;

namespace NeuralNetwork.Core
{
    public class Trainer
    {
        public NeuralNetwork? Train(NeuralNetwork network, double[][] inputs, double[][] targets, double minutesToTrain = 1.0)
        {
            const int epochs = 10000;//Amount of train session per activation fn + seed.
            NeuralNetwork? best = null;
            double bestValue = double.MaxValue;
            Task[] tasks = new Task[50];
            int[] seedSteps = new int[tasks.Length];
            int value = int.MinValue;
            for(int i = 0; i<seedSteps.Length; i++)
            {
                seedSteps[i] = value;
                value += (int)(uint.MaxValue / tasks.Length);
            }

            Stopwatch sw = Stopwatch.StartNew();

            //Create all the threads, each thread respect other seed ranges.
            for (int i = 0; i < tasks.Length; i++)
            {
                //Get the starting seed.
                int seed = seedSteps[i];

                tasks[i] = Task.Run(() =>
                {
                    int it = 0;//Iterations amount.

                    //While we still have time and the seed value is not collapse with other thread.
                    while (sw.Elapsed.TotalMinutes < minutesToTrain && it < (int)(uint.MaxValue / tasks.Length))
                    {

                        //Iterate all of the activation functions to find the best.
                        for(int act = 0; act < ActivationFunction.Functions.Length; act+= 2)
                        {
                            NeuralNetwork cp = new NeuralNetwork(network, seed);
                            cp.Activation = ActivationFunction.Functions[act];
                            cp.ActivationDerivative = ActivationFunction.Functions[act + 1];

                            //Train and get the test error result.
                            cp.Train(inputs, targets, epochs);
                            double testResult = cp.Test(inputs, targets);

                            //Check if is it the best yet.
                            if (testResult < bestValue)
                            {
                                bestValue = testResult;
                                best = cp;
                            }
                        }

                        it++;
                        seed++;
                    }
                });
            }
            Task.WaitAll(tasks);

            return best;
        }
    }
}
