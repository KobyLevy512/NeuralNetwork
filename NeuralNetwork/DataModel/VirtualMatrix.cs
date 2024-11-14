using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.DataModel
{
    public class VirtualMatrix
    {
        static int ids = 0;
        string path;
        int x , y;
        BinaryReader br;
        BinaryWriter bw;

        public int GetLength(int index)
        {
            if(index == 0)return y;
            return x;
        }

        public double this[int y, int x]
        {
            get => Read(y, x);
            set => Write(y, x, value);
        }
        public VirtualMatrix(int y, int x)
        {
            this.x = x;
            this.y = y;
            path = "virtulMatrix" + ids + ".mat";
            ids++;
            FileStream stream = File.Create(path);
            bw = new BinaryWriter(stream);
            br = new BinaryReader(stream);
            bw.Write(y);
            bw.Write(x);
            bw.Flush();
        }

        ~VirtualMatrix()
        {
            br.Close();
        }

        public void Write(int y, int x, double value)
        {
            bw.BaseStream.Position = 8 + (((long)y * this.x * 8) + ((long)x * 8));
            bw.Write(value);
            bw.Flush();
        }

        public double Read(int x, int y)
        {
            br.BaseStream.Position = 8 + (((long)y * this.x * 8) + ((long)x * 8));
            return br.ReadDouble();
        }
    }
}
