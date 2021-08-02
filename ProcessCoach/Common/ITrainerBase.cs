using System;
using Microsoft.ML.Data;

namespace ProcessCoach.Common
{
    public interface ITrainerBase
    {
        string Name { get; }
        void Fit(string trainingFileName);
        ClusteringMetrics Evaluate();
        void Save();
    }
}
