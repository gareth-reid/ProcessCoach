using System;
using Microsoft.ML.Data;

namespace ProcessCoach.DataModels
{
    public class MetricsPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId;

        [ColumnName("Score")]
        public float[] Distances;

        //public float ResultAccuracy;
        
    }
}
