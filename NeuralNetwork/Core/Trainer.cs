using NeuralNetwork.DataModel;
using System.Diagnostics;

namespace NeuralNetwork.Core
{
    public class Trainer
    {
        public class TrainResult
        {
            public int TotalSeeds;
            public int Threads;
            public Dictionary<int, int> SeedsByThread = new Dictionary<int, int>();
            public double BestValue;
            public double TotalTime;
            public int[] SeedsStart;

            public override string ToString()
            {
                string ret = "TotalSeed = " + TotalSeeds + "\n";
                ret += "Threads = " + Threads + "\n";
                ret += "BestValue = " + BestValue + "\n";
                ret += "TotalTime = " + TotalTime + "\n";
                ret += "SeedsByThread:\n";
                foreach(var key in SeedsByThread.Keys)
                {
                    ret += key + ":" + SeedsByThread[key] + "\n";
                }
                return ret;
            }
        }

        int threads, epochs;
        double minutesToTrain;
        TrainResult? result;

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

        public TrainResult? GetLastTrainResult()
        {
            if (result == null) return result;
            result.TotalSeeds = 0;
            foreach(var key in result.SeedsByThread.Keys)
            {
                result.TotalSeeds += result.SeedsByThread[key];
            }
            return result;
        }

        public void Interaction(Task[] threads, Stopwatch sw)
        {
            bool cancel = false;
            Func<bool> threadsCompelete = () =>
            {
                for (int i = 0; i < threads.Length; i++)
                {
                    if (!threads[i].IsCompleted) return false;
                }
                return true;
            };

            while(!cancel)
            {
                if (threadsCompelete()) return;
                Console.WriteLine("1:Print current Training results.");
                Console.WriteLine("2:Print total minutes pass.");
                Console.WriteLine("3:End Training Process.");
                string? input = Console.ReadLine();
                if (input == null) continue;
                if(input == "1")
                {
                    result.TotalTime = sw.Elapsed.TotalMinutes;
                    Console.WriteLine(result.ToString());
                }
                else if(input == "2")
                {
                    Console.WriteLine(sw.Elapsed.TotalMinutes.ToString("0.00"));
                }
                else if(input == "3")
                {
                    Console.WriteLine("Are you sure?(y,n)");
                    input = Console.ReadLine();
                    if(input == null) continue;
                    if (input == "y") return;

                }
                Console.WriteLine("Press Any Key to continue.");
                Console.ReadKey();
                Console.Clear();
            }
        }
        public BigDN? Train(BigDN network, DataStream stream, bool interaction = false, bool backup = false)
        {
            result = new TrainResult();
            result.Threads = threads;
            BigDN? best = null;
            double bestValue = double.MaxValue;
            Task[] tasks = new Task[threads];
            int[] seedSteps = BuildSeedSteps(threads);
            result.SeedsStart = seedSteps;
            Stopwatch sw = Stopwatch.StartNew();
            //Create all the threads, each thread respect other seed ranges.
            for (int i = 0; i < tasks.Length; i++)
            {
                //Get the starting seed.
                int seed = seedSteps[i];
                result.SeedsByThread.Add(i, 0);
                tasks[i] = Task.Run(() =>
                {
                    int it = 0;//Iterations amount.
                    //While we still have time and the seed value is not collapse with other thread.
                    while (sw.Elapsed.TotalMinutes < minutesToTrain && it < (int)(uint.MaxValue / tasks.Length))
                    {

                        //Iterate all of the activation functions to find the best.
                        for (int act = 0; act < ActivationFunction.Functions.Length; act += 2)
                        {
                            BigDN cp = new BigDN(network, seed);
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
                                result.BestValue = bestValue;
                                if(backup)
                                {
                                    best.SaveToFile("backup.model");
                                }
                            }
                        }

                        it++;
                        seed++;
                    }
                    result.SeedsByThread[i] = it;
                });
            }
            if(interaction)
            {
                Interaction(tasks, sw);
            }
            else
            {
                Task.WaitAll(tasks);
            }
            result.TotalTime = sw.Elapsed.TotalMinutes;
            return best;
        }
        public BigDN? Train(BigDN network, double[][] inputs, double[][] targets, bool interaction = false, bool backup = false)
        {
            result = new TrainResult();
            result.Threads = threads;
            BigDN? best = null;
            double bestValue = double.MaxValue;
            Task[] tasks = new Task[threads];
            int[] seedSteps = BuildSeedSteps(threads);
            Stopwatch sw = Stopwatch.StartNew();
            //Create all the threads, each thread respect other seed ranges.
            for (int i = 0; i < tasks.Length; i++)
            {
                //Get the starting seed.
                int seed = seedSteps[i];
                result.SeedsByThread.Add(i, 0);
                tasks[i] = Task.Run(() =>
                {
                    int it = 0;//Iterations amount.

                    //While we still have time and the seed value is not collapse with other thread.
                    while (sw.Elapsed.TotalMinutes < minutesToTrain && it < (int)(uint.MaxValue / tasks.Length))
                    {

                        //Iterate all of the activation functions to find the best.
                        for (int act = 0; act < ActivationFunction.Functions.Length; act += 2)
                        {
                            BigDN cp = new BigDN(network, seed);
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
                                result.BestValue = bestValue;
                                if (backup)
                                {
                                    best.SaveToFile("backup.model");
                                }
                            }
                        }

                        it++;
                        seed++;
                    }
                    result.SeedsByThread[i] = it;
                });
            }
            if (interaction)
            {
                Interaction(tasks, sw);
            }
            else
            {
                Task.WaitAll(tasks);
            }
            result.TotalTime = sw.Elapsed.TotalMinutes;
            return best;
        }
        public DeepNeural? Train(DeepNeural network, DataStream stream, bool interaction = false, bool backup = false)
        {
            result = new TrainResult();
            result.Threads = threads;
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
                result.SeedsByThread.Add(i, 0);
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
                                result.BestValue = bestValue;
                                if (backup)
                                {
                                    best.SaveToFile("backup.model");
                                }
                            }
                        }

                        it++;
                        seed++;
                    }
                    result.SeedsByThread[i] = it;
                });
            }
            if (interaction)
            {
                Interaction(tasks, sw);
            }
            else
            {
                Task.WaitAll(tasks);
            }
            result.TotalTime = sw.Elapsed.TotalMinutes;
            return best;
        }
        public DeepNeural? Train(DeepNeural network, double[][] inputs, double[][] targets, bool interaction = false, bool backup = false)
        {
            result = new TrainResult();
            result.Threads = threads;
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
                result.SeedsByThread.Add(i, 0);
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
                                result.BestValue = bestValue;
                                if (backup)
                                {
                                    best.SaveToFile("backup.model");
                                }
                            }
                        }

                        it++;
                        seed++;
                    }
                    result.SeedsByThread[i] = it;
                });
            }
            if (interaction)
            {
                Interaction(tasks, sw);
            }
            else
            {
                Task.WaitAll(tasks);
            }
            result.TotalTime = sw.Elapsed.TotalMinutes;
            return best;
        }
        public NeuralNetwork? Train(NeuralNetwork network, double[][] inputs, double[][] targets, bool interaction = false, bool backup = false)
        {
            result = new TrainResult();
            result.Threads = threads;
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
                result.SeedsByThread.Add(i, 0);
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
                                result.BestValue = bestValue;
                                if (backup)
                                {
                                    best.SaveToFile("backup.model");
                                }
                            }
                        }

                        it++;
                        seed++;
                    }
                    result.SeedsByThread[i] = it;
                });
            }
            if (interaction)
            {
                Interaction(tasks, sw);
            }
            else
            {
                Task.WaitAll(tasks);
            }
            result.TotalTime = sw.Elapsed.TotalMinutes;
            return best;
        }
    }
}