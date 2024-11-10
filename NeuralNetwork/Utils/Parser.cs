
namespace NeuralNetwork.Utils
{
    public unsafe static class Parser
    {
        public static double[] StringToDouble(string input)
        {
            double[] result = new double[input.Length];

            ushort[] asShort = new ushort[input.Length];
            for(int i = 0; i < input.Length; i++)
            {
                ushort a = input[i];
                ushort b = input[i+1];
                ushort c = input[i+2];
                ushort d = input[i+3];
                asShort[i] = input[i];
            }


            return result;
        }
    }
}
