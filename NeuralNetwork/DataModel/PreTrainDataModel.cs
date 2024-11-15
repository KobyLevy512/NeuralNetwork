
namespace NeuralNetwork.DataModel
{
    public abstract class PreTrainDataModel
    {
        public abstract (double[][] inputs, double[][] outputs) GetData();

        public class LogicGatesModel : PreTrainDataModel
        {
            public override (double[][] inputs, double[][] outputs) GetData()
            {
                var inputs = new double[12][]
                {
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1],
                    [2, 0, 0],
                    [2, 0, 1],
                    [2, 1, 0],
                    [2, 1, 1],
                };
                var outputs = new double[12][]
                {
                    [0],
                    [0],
                    [0],
                    [1],
                    [0],
                    [1],
                    [1],
                    [1],
                    [0],
                    [1],
                    [1],
                    [0],
                };
                return (inputs, outputs);
            }
        }
    }
}
