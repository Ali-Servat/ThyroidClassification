using OfficeOpenXml;
using System.Text;

class DataHandler
{
    public static double[][] ImportData(string path)
    {
        double[][] data;
        ExcelPackage.LicenseContext = LicenseContext.NonCommercial;
        using (ExcelPackage xlPackage = new(new FileInfo(path)))
        {
            var myWorksheet = xlPackage.Workbook.Worksheets.First();
            var totalRows = myWorksheet.Dimension.End.Row;
            var totalColumns = myWorksheet.Dimension.End.Column;

            var sb = new StringBuilder();
            data = new double[totalColumns][];

            for (int i = 0; i < data.Length; i++)
            {
                data[i] = new double[totalRows];
            }

            for (int colNum = 1; colNum <= totalColumns; colNum++)
            {
                for (int rowNum = 1; rowNum <= totalRows; rowNum++)
                {
                    data[colNum - 1][rowNum - 1] = Convert.ToDouble(myWorksheet.Cells[rowNum, colNum].Value);
                }
            }
        }
        return data;
    }

    public static void ShuffleRows(double[][] array1, double[][] array2)
    {
        int rows = array1.Length;
        int columns1 = array1.First().Length;
        int columns2 = array2.First().Length;

        Random rand = new Random();
        int[] rowIndices = Enumerable.Range(0, rows).OrderBy(x => rand.Next()).ToArray();

        double[][] shuffledArray1 = new double[rows][];
        double[][] shuffledArray2 = new double[rows][];

        for (int i = 0; i < rows; i++)
        {
            shuffledArray1[i] = new double[columns1];
            shuffledArray2[i] = new double[columns2];
        }

        for (int newRow = 0; newRow < rows; newRow++)
        {
            int originalRow = rowIndices[newRow];

            for (int col1 = 0; col1 < columns1; col1++)
                shuffledArray1[newRow][col1] = array1[originalRow][col1];

            for (int col2 = 0; col2 < columns2; col2++)
                shuffledArray2[newRow][col2] = array2[originalRow][col2];
        }

        Array.Copy(shuffledArray1, array1, array1.Length);
        Array.Copy(shuffledArray2, array2, array2.Length);
    }
    public static (double[][] i1, double[][] i2, double[][] i3, double[][] t1, double[][] t2, double[][] t3) SplitData(double[][] inputs, double[][] targets, double validationDataRatio, double testDataRatio)
    {
        int validationDataCount = (int)(validationDataRatio * targets.Length);
        int testDataCount = (int)(testDataRatio * targets.Length);
        int trainingDataCount = targets.Length - validationDataCount - testDataCount;

        double[][] trainingInputs = new double[trainingDataCount][];
        double[][] validationInputs = new double[validationDataCount][];
        double[][] testInputs = new double[testDataCount][];

        double[][] trainingTargets = new double[trainingDataCount][];
        double[][] validationTargets = new double[validationDataCount][];
        double[][] testTargets = new double[testDataCount][];

        for (int i = 0; i < trainingDataCount; i++)
        {
            trainingInputs[i] = new double[inputs.First().Length];
            trainingTargets[i] = new double[targets.First().Length];
        }

        for (int i = 0; i < validationDataCount; i++)
        {
            validationInputs[i] = new double[inputs.First().Length];
            validationTargets[i] = new double[targets.First().Length];
        }

        for (int i = 0; i < testDataCount; i++)
        {
            testInputs[i] = new double[inputs.First().Length];
            testTargets[i] = new double[targets.First().Length];
        }

        Array.Copy(inputs, trainingInputs, trainingDataCount);
        Array.Copy(inputs, trainingDataCount, validationInputs, 0, validationDataCount);
        Array.Copy(inputs, trainingDataCount + validationDataCount, testInputs, 0, testDataCount);

        Array.Copy(targets, trainingTargets, trainingDataCount);
        Array.Copy(targets, trainingDataCount, validationTargets, 0, validationDataCount);
        Array.Copy(targets, trainingDataCount + validationDataCount, testTargets, 0, testDataCount);

        return (trainingInputs, validationInputs, testInputs, trainingTargets, validationTargets, testTargets);
    }
}