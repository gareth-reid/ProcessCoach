using Microsoft.ML;
using Microsoft.ML.Trainers;
using ProcessCoach.Common;

namespace ProcessCoach.Trainers
{
    public class KMeansTrainer : TrainerBase<KMeansModelParameters>
    {
        public KMeansTrainer(int numberOfClusters) : base()
        {
            Name = $"K Means Clulstering - {numberOfClusters} Clusters";
            _model = MlContext.Clustering.Trainers
                  .KMeans(numberOfClusters: numberOfClusters, featureColumnName: "Features");
        }
    }
}