
using NeuralNetwork.Utils;
using System.Reflection;

namespace NeuralNetwork.DataModel
{
    public abstract class DataModelBase
    {
        [AttributeUsage(AttributeTargets.Field)]
        public class FieldAttribute : System.Attribute
        {
            public int Location;
            public Type Type;
            public bool Input;
            public int Size;
            public FieldAttribute(int location, Type type, bool input, int size = 0)
            {
                this.Type = type;
                this.Location = location;
                Input = input;
                Size = size;
            }
        }

        protected int inputs, outputs;
        FieldInfo[] inputsFields, targetsFields;
        public DataModelBase(int inputs, int outputs)
        {
            this.inputs = inputs;
            this.outputs = outputs;

            inputsFields = 
                GetType().
                GetFields().
                Where(f => f.GetCustomAttribute<FieldAttribute>() != null && f.GetCustomAttribute<FieldAttribute>().Input).
                OrderBy(f => f.GetCustomAttribute<FieldAttribute>()?.Location).
                ToArray();

            targetsFields =
                GetType().
                GetFields().
                Where(f => f.GetCustomAttribute<FieldAttribute>() != null && !f.GetCustomAttribute<FieldAttribute>().Input).
                OrderBy(f => f.GetCustomAttribute<FieldAttribute>()?.Location).
                ToArray();
        }
        double[] GetFieldValues(FieldInfo[] fields)
        {
            List<double> ret = new List<double>();
            foreach (var field in fields)
            {
                FieldAttribute? att = field.GetCustomAttribute<FieldAttribute>();
                if (att == null) continue;
                Type type = att.Type;
                if (type == typeof(string))
                {
                    string? value = field.GetValue(this)?.ToString();
                    if (value == null) continue;
                    int size = att.Size;
                    if (value.Length > size)
                    {
                        value = value.Substring(0, size);
                    }
                    else
                    {
                        while (value.Length < size)
                        {
                            value += ((char)0).ToString();
                        }
                    }
                    ret.AddRange(Parser.StringToDouble(value));
                }
                else if (type == typeof(int))
                {
                    int? value = (int?)field.GetValue(this);
                    if (value == null) continue;
                    ret.Add((double)value);
                }
                else if (type == typeof(double))
                {
                    double? value = (double?)field.GetValue(this);
                    if (value == null) continue;
                    ret.Add((double)value);
                }
                else if (type == typeof(bool))
                {
                    bool? value = (bool?)field.GetValue(this);
                    if (value == null) continue;
                    ret.Add(value == true ? 1 : 0);
                }
            }

            return ret.ToArray();
        }
        public double[] GetTarget()
        {
            return GetFieldValues(targetsFields);
        }
        public double[] GetInput()
        {
            return GetFieldValues(inputsFields);
        }
    }
}
