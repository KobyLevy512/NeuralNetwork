
using System.Runtime.InteropServices;

namespace NeuralNetwork.DataModel
{
    public unsafe class VirtualMatrix
    {
        static int ids = 0;
        int x, y;
        byte[] buffer;
        FileStream stream;

        public int GetLength0
        {
            get => y;
        }

        public int GetLength1
        {
            get => x;
        }

        public VirtualMatrix(int y, int x)
        {
            stream = new FileStream("p", FileMode.CreateNew, FileAccess.ReadWrite, FileShare.ReadWrite, x * 8);
            buffer = new byte[x * 8];
        }

        public double* ReadRow(int index)
        {
            stream.Position = index * x * 8;
            stream.Read(buffer, 0, buffer.Length);
            fixed(byte* ptr = &buffer[0])
            {
                double* asDouble = (double*)ptr;
                return asDouble;
            }
        }

        public void WriteRow(int index)
        {
            stream.Position = index * x * 8;
            stream.Write(buffer, 0, buffer.Length);
            stream.Flush();
        }
    }
}
