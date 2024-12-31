class MLP
{
    private double[][][] _weights;
    private double[][] _biases;
    private Neuron[][] _neurons;
    private double[][] _deltaFactors;
    public double MSE { get; private set; }
    private Random _r;

    public MLP(int[] layers)
    {
        _r = new Random();
        _neurons = new Neuron[layers.Length][];

        for (int i = 0; i < layers.Length; i++)
        {
            _neurons[i] = new Neuron[layers[i]];
        }

        _weights = new double[layers.Length - 1][][];
        for (int i = 0; i < _weights.Length; i++)
        {
            _weights[i] = new double[_neurons[i].Length][];
            for (int j = 0; j < _neurons[i].Length; j++)
            {
                _weights[i][j] = new double[_neurons[i + 1].Length];
            }
        }

        _biases = new double[_weights.Length][];
        for (int i = 0; i < _biases.Length; i++)
        {
            _biases[i] = new double[_neurons[i + 1].Length];
        }

        _deltaFactors = new double[_neurons.Length][];
        for (int i = 0; i < _deltaFactors.Length; i++)
        {
            _deltaFactors[i] = new double[_neurons[i].Length];
        }
    }

    public void Train(double[][] trainingData, double[][] trainingTargets, double[][] validationData, double[][] validationTargets, int epochs, double learningRate)
    {
        InitializeWeights();
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int sample = 0; sample < trainingData.Length; sample++)
            {
                FeedForward(trainingData[sample]);

                double[] currentTargets = trainingTargets[sample];
                BackPropagate(currentTargets, learningRate);
            }
            Validate(validationData, validationTargets);
            Console.WriteLine($"Epoch {epoch + 1} ended. MSE: {MSE}");
        }

    }

    private void Validate(double[][] validationInputs, double[][] validationTargets)
    {
        MSE = 0;

        for (int i = 0; i < validationInputs.Length; i++)
        {
            FeedForward(validationInputs[i]);

            double squaredError = 0;
            for (int k = 0; k < validationTargets.First().Length; k++)
            {
                double predictedValue = _neurons[^1][k].ActivityLevel;
                squaredError += Math.Pow(validationTargets[i][k] - predictedValue, 2);
            }
            squaredError /= validationTargets.First().Length;
            MSE += squaredError;
        }
        MSE /= validationTargets.Length;
    }

    private void BackPropagate(double[] targets, double learningRate)
    {
        for (int i = 0; i < _neurons[^1].Length; i++)
        {
            double error = targets[i] - _neurons[^1][i].ActivityLevel;
            MSE += Math.Pow(error, 2);
            double deltaFactor = error * ActivationFunctionDerivative(_neurons[^1][i].netInput);
            _deltaFactors[^1][i] = deltaFactor;
        }

        for (int i = _neurons.Length - 2; i >= 0; i--)
        {
            for (int j = 0; j < _neurons[i].Length; j++)
            {
                double netDeltaInput = CalculateNetDeltaInput(i, j);
                double deltaFactor = netDeltaInput * ActivationFunctionDerivative(_neurons[i][j].netInput);
                _deltaFactors[i][j] = deltaFactor;
            }
        }

        for (int i = 0; i < _weights.Length; i++)
        {
            for (int j = 0; j < _weights[i].Length; j++)
            {
                for (int k = 0; k < _weights[i][j].Length; k++)
                {
                    _weights[i][j][k] += learningRate * _deltaFactors[i + 1][k] * _neurons[i][j].ActivityLevel;
                    _biases[i][k] += learningRate * _deltaFactors[i + 1][k];
                }
            }
        }
    }
    private void InitializeWeights()
    {
        for (int i = 0; i < _weights.Length; i++)
        {
            for (int j = 0; j < _weights[i].Length; j++)
            {
                for (int k = 0; k < _weights[i][j].Length; k++)
                {
                    _weights[i][j][k] = _r.NextDouble() - 0.5;
                    _biases[i][k] = _r.NextDouble() - 0.5;
                }
            }
        }
    }
    private void FeedForward(double[] input)
    {
        for (int i = 0; i < _neurons[0].Length; i++)
        {
            _neurons[0][i].ActivityLevel = input[i];
        }

        for (int i = 1; i < _neurons.Length; i++)
        {
            for (int j = 0; j < _neurons[i].Length; j++)
            {
                double netInput = CalculateNetInput(i, j);
                _neurons[i][j].netInput = netInput;
                _neurons[i][j].ActivityLevel = ActivationFunction(netInput);
            }
        }
    }
    public int Classify(double[] input)
    {
        FeedForward(input);
        return _neurons[^1].Select((value, index) => new { Index = index, Value = value.ActivityLevel }).OrderByDescending(x => x.Value).First().Index;

    }
    private double CalculateNetInput(int layerIndex, int memberIndex)
    {
        double sum = 0;
        Neuron[] previousLayer = _neurons[layerIndex - 1];

        for (int i = 0; i < previousLayer.Length; i++)
        {
            sum += previousLayer[i].ActivityLevel * _weights[layerIndex - 1][i][memberIndex];
        }
        sum += _biases[layerIndex - 1][memberIndex];

        return sum;
    }
    private double CalculateNetDeltaInput(int layerIndex, int memberIndex)
    {
        double sum = 0;

        for (int i = 0; i < _neurons[layerIndex + 1].Length; i++)
        {
            sum += _deltaFactors[layerIndex + 1][i] * _weights[layerIndex][memberIndex][i];
        }
        return sum;
    }
    private double ActivationFunction(double netInput)
    {
        return (2.0 / (1 + Math.Pow(Math.E, -netInput))) - 1;
    }
    private double ActivationFunctionDerivative(double x)
    {
        return 1.0 / 2 * (1 + ActivationFunction(x)) * (1 - ActivationFunction(x));
    }

}