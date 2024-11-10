
namespace NeuralNetwork.DataModel
{
    internal class QuestionAnswerModel : DataModelBase
    {
        [Field(0, true, 16)]
        public string Question;

        [Field(0, false, 16)]
        public string Answer;

        public QuestionAnswerModel(string question, string answer) : base(1, 1)
        {
            Question = question;
            Answer = answer;
        }
    }
}
