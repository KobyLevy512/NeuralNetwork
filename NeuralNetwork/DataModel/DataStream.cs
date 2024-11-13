
namespace NeuralNetwork.DataModel
{
    public class DataStream
    {
        BinaryReader br;
        int count;
        int inputs, outputs;
        string path;

        /// <summary>
        /// The count of entries in the stream.
        /// </summary>
        public int Count
        {
            get => count;
        }

        public DataStream(DataStream stream)
        {
            inputs = stream.inputs;
            outputs = stream.outputs;
            path = stream.path;
            br = new BinaryReader(File.OpenRead(path));
            count = br.ReadInt32();
        }
        public DataStream(int inputs, int outputs, string filePath)
        {
            this.path = filePath;
            this.inputs = inputs;
            this.outputs = outputs;
            br = new BinaryReader(File.OpenRead(path));
            count = br.ReadInt32();
        }

        public (double[], double[]) ReadEntry(int index)
        {
            br.BaseStream.Position = 4 + (index * ((inputs * 8) + (outputs * 8)));

            var ret = (new double[inputs], new double[outputs]);

            for(int i = 0; i < inputs; i++)
            {
                ret.Item1[i] = br.ReadDouble();
            }

            for(int i = 0; i < outputs; i++)
            {
                ret.Item2[i] = br.ReadDouble();
            }
            return ret;
        }
        public void Close()
        {
            br.Close();
        }
    }
}
