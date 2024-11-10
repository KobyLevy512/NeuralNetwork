
namespace NeuralNetwork.Core
{
    public static class ActivationFunction
    {
        public delegate double Activation(double x);

        public static Activation[] Functions = new Activation[14]
        {
            Sigmoid,
            SigmoidDerivative,
            ReLU,
            ReLUDerivative,
            LeakyReLU,
            LeakyReLUDerivative,
            Tanh,
            TanhDerivative,
            Swish,
            SwishDerivative,
            ELU,
            ELUDerivative,
            GELU,
            GELUDerivative
        };
        public static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
        public static double SigmoidDerivative(double x) => x * (1 - x);
        public static double ReLU(double x) => Math.Max(0, x);
        public static double ReLUDerivative(double x) => x > 0 ? 1 : 0;
        public static double LeakyReLU(double x) => x > 0 ? x : 0.01 * x; //0.01 can change
        public static double LeakyReLUDerivative(double x) => x > 0 ? 1 : 0.01; //0.01 can change
        public static double Tanh(double x) => Math.Tanh(x);
        public static double TanhDerivative(double x)
        {
            double tanh = Tanh(x);
            return 1 - tanh * tanh;
        }
        public static double Swish(double x) => x * Sigmoid(x);
        public static double SwishDerivative(double x)
        {
            double sigmoid = Sigmoid(x);
            return sigmoid + x * sigmoid * (1 - sigmoid);
        }
        public static double ELU(double x) => x >= 0 ? x : 1.0 * (Math.Exp(x) - 1); //1.0 can change
        public static double ELUDerivative(double x) => x >= 0 ? 1 : 1.0 * Math.Exp(x); //1.0 can change
        public static double GELU(double x) =>
             0.5 * x * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (x + 0.044715 * Math.Pow(x, 3))));

        public static double GELUDerivative(double x)
        {
            double tanhApprox = Math.Tanh(Math.Sqrt(2 / Math.PI) * (x + 0.044715 * Math.Pow(x, 3)));
            double factor = 0.5 * (1 + tanhApprox) + 0.5 * x * (1 - Math.Pow(tanhApprox, 2)) * (Math.Sqrt(2 / Math.PI) * (1 + 3 * 0.044715 * Math.Pow(x, 2)));
            return factor;
        }
    }
}
