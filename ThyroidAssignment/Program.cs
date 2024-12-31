using ThyroidAssignment;

double[][] inputs = DataHandler.ImportData("Data/ThyroidInputs.xlsx");
double[][] targets = DataHandler.ImportData("Data/ThyroidTargets.xlsx");

while (true)
{
    Console.Clear();
    Console.WriteLine("Thyroid Assignment \n \n");

    DataHandler.ShuffleRows(inputs, targets);

    var data = DataHandler.SplitData(inputs, targets, 0.15, 0.15);
    double[][] trainingInputs = data.i1;
    double[][] validationInputs = data.i2;
    double[][] testInputs = data.i3;
    double[][] trainingTargets = data.t1;
    double[][] validationTargets = data.t2;
    double[][] testTargets = data.t3;

    int epochs = (int)GetDouble("Enter the number of epochs: ");
    double learningRate = GetDouble("Enter learning rate: ");
    int[] layers = GetLayers(trainingInputs.First().Length, targets.First().Length);


    MLP mlp = new(layers);
    mlp.Train(trainingInputs, trainingTargets, validationInputs, validationTargets, epochs, learningRate);

    int[] classificationResults = new int[testTargets.Length];
    for (int i = 0; i < testInputs.Length; i++)
    {
        int classificationIndex = mlp.Classify(testInputs[i]);
        classificationResults[i] = classificationIndex;
    }

    int[] targetIndices = new int[testTargets.Length];
    for (int i = 0; i < testTargets.Length; i++)
    {
        int targetIndex = testTargets[i].Select((value, index) => new { Value = value, Index = index }).OrderByDescending(x => x.Value).First().Index;
        targetIndices[i] = targetIndex;
    }

    int[,] confusionMatrix = Evaluator.ConstructConfusionMatrix(targetIndices, classificationResults);
    double[,] evaluation = Evaluator.Evaluate(confusionMatrix);

    Evaluator.PrintConfusionMatrix(confusionMatrix);
    Evaluator.PrintEvaluationTable(evaluation);
    Evaluator.PrintMacroScores(evaluation);

    Console.Write("Press any key to restart the program: ");
    Console.ReadKey();
}

static double GetDouble(string message)
{
    bool shouldStop = false;
    double result = double.NaN;

    while (!shouldStop)
    {
        Console.Write(message);
        string? input = Console.ReadLine();
        shouldStop = double.TryParse(input, out result);
    }
    return result;
}

static int[] GetLayers(int inputSize, int outputSize)
{
    bool shouldStop = false;
    string? input = string.Empty;

    while (!shouldStop)
    {
        Console.Write("Enter hidden layers. example: 4 5 (first hidden layer with 4 neurons and second hidden layer with 5 neurons): ");
        input = Console.ReadLine();
        shouldStop = input != null && input != string.Empty;
    }
    input = input.Trim();
    int[] hiddenLayers = input.Split(' ').Select(x => int.Parse(x)).ToArray();
    int[] layers = [inputSize, .. hiddenLayers, outputSize];

    return layers;
}