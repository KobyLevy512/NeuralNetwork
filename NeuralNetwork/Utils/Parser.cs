
namespace NeuralNetwork.Utils
{
    public unsafe static class Parser
    {
        static string[] SplitStringBySize(string input, int len)
        {
            string[] ret;
            if (input.Length % len == 0)
            {
                ret = new string[input.Length / len];
            }
            else
            {
                ret = new string[input.Length / len + 1];
            }
            int retIndex = 0;
            foreach(char c in input)
            {
                ret[retIndex] += c.ToString();
                if (ret[retIndex].Length == len)
                    retIndex++;
            }
            if (retIndex == ret.Length) return ret;
            while(ret[retIndex].Length < len)
            {
                ret[retIndex] += ((char)0).ToString();
            }
            return ret;
            
        }
        public static double[] StringToDouble(string input)
        {
            string[] split = SplitStringBySize(input, 4);
            double[] result = new double[split.Length];
            for(int i = 0; i < split.Length; i++)
            {
                long val = split[i][0] | ((long)split[i][1] << 16) | ((long)split[i][2] << 32) | ((long)split[i][3] << 48);
                double* ptr = (double*)&val;
                result[i] = *ptr;
            }
            return result;
        }

        public static string DoubleToString(double[] values)
        {
            string ret = "";
            foreach(double d in values)
            {
                long* ptr = (long*)&d;
                long value = *ptr;
                for(int i = 0; i < 4; i++)
                {
                    ret += ((char)value).ToString();
                    value >>= 16;
                }
            }
            return ret;
        }
    }
}
