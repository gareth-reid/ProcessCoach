using System;
using System.Collections.Generic;
using System.Diagnostics;
using NUnit.Framework;
using ProcessCoach.Common;
using ProcessCoach.DataModels;
using ProcessCoach.Predictors;
using ProcessCoach.Trainers;

namespace ProcessCoach.Tests
{
    public class Tests
    {
        [SetUp]
        public void Setup()
        {            
        }

        [Test]
        public void Test1()
        {
            var newSample = new DnaQcData
            {
                WhiteCellCount = 5000f,
                InstrumentTempreture = 25.2f,
                Volume = 79.3f,
                //Concentration = 100,
                SampleType = "DNA"
            };


            var trainers = new List<ITrainerBase>
            {
                new KMeansTrainer(1),
                new KMeansTrainer(2),
                new KMeansTrainer(3),
                new KMeansTrainer(4),
                new KMeansTrainer(5),
            };

            trainers.ForEach(t => TrainEvaluatePredict(t, newSample));
        }

        static void TrainEvaluatePredict(ITrainerBase trainer, DnaQcData newSample)
        {
            TestContext.Out.WriteLine("*******************************");
            TestContext.Out.WriteLine($"{ trainer.Name }");
            TestContext.Out.WriteLine("*******************************");

            trainer.Fit("/Users/reid gareth/Projects/ProcessCoach/TrainingData/RandomDnaQCMetrics.csv");

            var modelMetrics = trainer.Evaluate();

            TestContext.Out.WriteLine($"Average Distance: {modelMetrics.AverageDistance:#.##}{Environment.NewLine}" +
                              $"Davies Bouldin Index: {modelMetrics.DaviesBouldinIndex:#.##}{Environment.NewLine}" +
                              $"Normalized Mutual Information: {modelMetrics.NormalizedMutualInformation:#.##}{Environment.NewLine}");

            trainer.Save();

            var predictor = new Predictor();
            var prediction = predictor.Predict(newSample);
            TestContext.Out.WriteLine("------------------------------");
            TestContext.Out.WriteLine($"Prediction: {prediction.PredictedClusterId:#.##}");
            TestContext.Out.WriteLine($"Distances: {string.Join(" ", prediction.Distances)}");
            //TestContext.Out.WriteLine($"Conc: {prediction.Concentration}");
            TestContext.Out.WriteLine("------------------------------");
        }
    }
}
    