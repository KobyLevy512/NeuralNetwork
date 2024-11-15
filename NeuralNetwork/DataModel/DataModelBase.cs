﻿
using System.Reflection;

namespace NeuralNetwork.DataModel
{
    public abstract class DataModelBase
    {
        [AttributeUsage(AttributeTargets.Field)]
        public class FieldAttribute : System.Attribute
        {
            public int Location;
            public bool Input;
            public int Size;
            public FieldAttribute(int location, bool input, int size = 0)
            {
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
                Type type = field.FieldType;
                if (type == typeof(int))
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
        int GetSize(FieldInfo[] fields)
        {
            int size = 0;
            foreach (var field in fields)
            {
                if (field.FieldType == typeof(string))
                {
                    int s = field.GetCustomAttribute<FieldAttribute>().Size;
                    size += s / 4;
                    if (s % 4 != 0)
                        size++;
                }
                else
                {
                    size++;
                }
            }
            return size;
        }
        public double[] GetTarget()
        {
            return GetFieldValues(targetsFields);
        }
        public double[] GetInput()
        {
            return GetFieldValues(inputsFields);
        }
        public int GetInputsSize()
        {
            return GetSize(inputsFields);
        }
        public int GetTargetsSize()
        {
            return GetSize(targetsFields);
        }
    }
}
