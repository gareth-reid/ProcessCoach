using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using ProcessCoach.DataModels;
using System;
using System.IO;

namespace ProcessCoach.Common

{
    /// <summary>
    /// Base class for Trainers.
    /// This class exposes methods for training, evaluating and saving ML Models.
    /// Classes that inherit this class need to assing concrete model and name.
    /// </summary>
    public abstract class TrainerBase<TParameters> : ITrainerBase
        where TParameters : class
    {
        public string Name { get; protected set; }

        protected static string ModelPath => Path.Combine(AppContext.BaseDirectory, "cluster.mdl");

        protected readonly MLContext MlContext;

        protected DataOperationsCatalog.TrainTestData _dataSplit;
        protected ITrainerEstimator<ClusteringPredictionTransformer<TParameters>, TParameters>
                                                      _model;
        protected ITransformer _trainedModel;

        protected TrainerBase()
        {
            MlContext = new MLContext(111);
        }

        /// <summary>
        /// Train model on defined data.
        /// </summary>
        /// <param name="trainingFileName"></param>
        public void Fit(string trainingFileName)
        {
            if (!File.Exists(trainingFileName))
            {
                throw new FileNotFoundException($"File {trainingFileName} doesn't exist.");
            }

            _dataSplit = LoadAndPrepareData(trainingFileName);
            var dataProcessPipeline = BuildDataProcessingPipeline();
            var trainingPipeline = dataProcessPipeline
                                    .Append(_model);

            _trainedModel = trainingPipeline.Fit(_dataSplit.TrainSet);
        }

        /// <summary>
        /// Evaluate trained model.
        /// </summary>
        /// <returns>Metrics object which contain information about model performance.</returns>        
        public ClusteringMetrics Evaluate()
        {
            var testSetTransform = _trainedModel.Transform(_dataSplit.TestSet);

            return MlContext.Clustering.Evaluate(
                data: testSetTransform,
                labelColumnName: "PredictedLabel",
                scoreColumnName: "Score",
                featureColumnName: "Features");
        }

        /// <summary>
        /// Save Model in the file.
        /// </summary>
        public void Save()
        {
            MlContext.Model.Save(_trainedModel, _dataSplit.TrainSet.Schema, ModelPath);
        }

        /// <summary>
        /// Feature engeneering and data pre-processing.
        /// </summary>
        /// <returns>Data Processing Pipeline.</returns>
        private EstimatorChain<ColumnConcatenatingTransformer> BuildDataProcessingPipeline()
        {
            var dataProcessPipeline =
                MlContext.Transforms.Text
                    .FeaturizeText(inputColumnName: "SampleType", outputColumnName: "SampleTypeFeaturized")                    
                .Append(MlContext.Transforms.Concatenate("Features",                                               
                                               nameof(DnaQcData.WhiteCellCount),
                                               nameof(DnaQcData.Volume),
                                               nameof(DnaQcData.InstrumentTempreture),
                                               nameof(DnaQcData.Concentration),
                                               nameof(DnaQcData.ResultAccuracy),
                                               "SampleTypeFeaturized"))
               .AppendCacheCheckpoint(MlContext);

            return dataProcessPipeline;
        }

        private DataOperationsCatalog.TrainTestData LoadAndPrepareData(string trainingFileName)
        {
            var trainingDataView = MlContext.Data.LoadFromTextFile<DnaQcData>(
                              trainingFileName,
                              hasHeader: true,
                              separatorChar: ',');
            return MlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.3);
        }
    }
}