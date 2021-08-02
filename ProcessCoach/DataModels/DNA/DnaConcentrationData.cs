using System;
using Microsoft.ML.Data;

namespace ProcessCoach.DataModels
{
    public class DnaQcData
    {
        [LoadColumn(0)]
        public float Volume { get; set; }
        [LoadColumn(1)]
        public float WhiteCellCount { get; set; }
        [LoadColumn(2)]
        public float InstrumentTempreture { get; set; }
        [LoadColumn(3)]
        public float Concentration { get; set; }
        [LoadColumn(4)]
        public string SampleType { get; set; }
        [LoadColumn(5)]
        public float ResultAccuracy { get; set; } //this is the target attribute
    }
}
