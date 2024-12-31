using ConsoleTables;

namespace ThyroidAssignment
{
    public class Evaluator
    {
        public static int[,] ConstructConfusionMatrix(int[] testTargets, int[] predictions)
        {
            var confusionMatrix = new int[3, 3];

            for (int i = 0; i < testTargets.Length; i++)
            {
                var target = testTargets[i];
                var prediction = predictions[i];

                confusionMatrix[target, prediction]++;
            }

            return confusionMatrix;
        }
        public static void PrintConfusionMatrix(int[,] confusionMatrix)
        {
            ConsoleTable table = new("actual/prediction", "sub function", "hyper function", "normal");

            for (int i = 0; i < confusionMatrix.GetLength(0); i++)
            {
                string label = i == 0 ? "sub function" : i == 1 ? "hyper function" : "normal";
                table.AddRow(label, confusionMatrix[i, 0], confusionMatrix[i, 1], confusionMatrix[i, 2]);
            }

            table.Configure((x) => x.EnableCount = false);
            Console.WriteLine("Confusion matrix:");
            table.Write();
        }
        public static double[,] Evaluate(int[,] confusionMatrix)
        {
            double[,] output = new double[confusionMatrix.GetLength(0), 4];

            int totalCount = 0;
            for (int i = 0; i < confusionMatrix.GetLength(0); i++)
            {
                for (int j = 0; j < confusionMatrix.GetLength(1); j++)
                {
                    totalCount += confusionMatrix[i, j];
                }
            }

            for (int i = 0; i < confusionMatrix.GetLength(0); i++)
            {
                double tpCount = 0;
                double tnCount = 0;
                double fpCount = 0;
                double fnCount = 0;

                int tpIndex = i;
                tpCount = confusionMatrix[i, tpIndex];

                for (int j = 0; j < confusionMatrix.GetLength(1); j++)
                {
                    fpCount = j == i ? fpCount : fpCount + confusionMatrix[i, j];
                }

                for (int j = 0; j < confusionMatrix.GetLength(0); j++)
                {
                    fnCount = j == i ? fnCount : fnCount + confusionMatrix[j, i];
                }

                tnCount = totalCount - tpCount - fnCount - fpCount;
                output[i, 0] = tpCount / (tpCount + fpCount);
                output[i, 1] = (tpCount + fnCount == 0) ? 0 : tpCount / (tpCount + fnCount);
                output[i, 2] = (tpCount + tnCount) / totalCount;
                output[i, 3] = (output[i, 0] + output[i, 1] == 0) ? 0 : (2 * output[i, 0] * output[i, 1]) / (output[i, 0] + output[i, 1]);
            }
            return output;
        }
        public static void PrintEvaluationTable(double[,] evaluationMatrix)
        {
            ConsoleTable table = new("", "Precision", "Recall", "Accuracy", "F1 Score");

            for (int i = 0; i < evaluationMatrix.GetLength(0); i++)
            {
                var precision = (evaluationMatrix[i, 0] * 100).ToString("F2") + "%";
                var recall = (evaluationMatrix[i, 1] * 100).ToString("F2") + "%";
                var accuracy = (evaluationMatrix[i, 2] * 100).ToString("F2") + "%";
                var f1Score = (evaluationMatrix[i, 3] * 100).ToString("F2") + "%";

                string label = i == 0 ? "sub function" : i == 1 ? "hyper function" : "normal";
                table.AddRow(label, precision, recall, accuracy, f1Score);
            }

            table.Configure((x) => x.EnableCount = false);

            Console.WriteLine("Evaluation table:");
            table.Write();
        }
        public static void PrintMacroScores(double[,] evaluationMatrix)
        {
            double[] MacroScores = new double[evaluationMatrix.GetLength(1)];
            for (int i = 0; i < evaluationMatrix.GetLength(1); i++)
            {
                double weightedAverage = 0;
                for (int j = 0; j < evaluationMatrix.GetLength(0); j++)
                {
                    weightedAverage += evaluationMatrix[j, i];
                }
                weightedAverage /= evaluationMatrix.GetLength(0);
                MacroScores[i] = weightedAverage;
            }

            var macroPrecision = (MacroScores[0] * 100).ToString("F2") + "%";
            var macroRecall = (MacroScores[1] * 100).ToString("F2") + "%";
            var macroAccuracy = (MacroScores[2] * 100).ToString("F2") + "%";
            var macroF1Score = (MacroScores[3] * 100).ToString("F2") + "%";

            Console.WriteLine($"Macro Precision: {macroPrecision}");
            Console.WriteLine($"Macro Recall: {macroRecall}");
            Console.WriteLine($"Macro Accuracy: {macroAccuracy}");
            Console.WriteLine($"Macro F1 Score: {macroF1Score}");
        }
    }
}
