using NeuralNetwork.DataModel;
using System.Diagnostics;

namespace NeuralNetwork.Core
{
    public class Trainer
    {
        int threads, epochs;
        double minutesToTrain;
        public Trainer(double minutesToTrain, int thread, int epochs)
        {
            this.minutesToTrain = minutesToTrain;
            this.threads = thread;
            this.epochs = epochs;
        }
        private int[] BuildSeedSteps(int size)
        {
            int[] seedSteps = new int[size];
            int value = int.MinValue;
            for (int i = 0; i < seedSteps.Length; i++)
            {
                seedSteps[i] = value;
                value += (int)(uint.MaxValue / size);
            }
            return seedSteps;
        }

        public DeepNeural? Train(DeepNeural network, DataStream stream)
        {
            DeepNeural? best = null;
            double bestValue = double.MaxValue;
            Task[] tasks = new Task[threads];
            int[] seedSteps = BuildSeedSteps(threads);
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
                        for (int act = 0; act < ActivationFunction.Functions.Length; act += 2)
                        {
                            DeepNeural cp = new DeepNeural(network, seed);
                            cp.Activation = ActivationFunction.Functions[act];
                            cp.ActivationDerivative = ActivationFunction.Functions[act + 1];

                            //Train and get the test error result.
                            cp.Train(stream, epochs);
                            double testResult = cp.Test(stream);

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
        public DeepNeural? Train(DeepNeural network, double[][] inputs, double[][] targets)
        {
            DeepNeural? best = null;
            double bestValue = double.MaxValue;
            Task[] tasks = new Task[threads];
            int[] seedSteps = BuildSeedSteps(threads);
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
                        for (int act = 0; act < ActivationFunction.Functions.Length; act += 2)
                        {
                            DeepNeural cp = new DeepNeural(network, seed);
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
        public NeuralNetwork? Train(NeuralNetwork network, double[][] inputs, double[][] targets)
        {
            NeuralNetwork? best = null;
            double bestValue = double.MaxValue;
            Task[] tasks = new Task[threads];
            int[] seedSteps = BuildSeedSteps(tasks.Length);
            int value = int.MinValue;
            for (int i = 0; i < seedSteps.Length; i++)
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
                        for (int act = 0; act < ActivationFunction.Functions.Length; act += 2)
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