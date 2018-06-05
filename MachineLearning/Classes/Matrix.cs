using System;
using System.Collections.Generic;
using System.Linq;

namespace MachineLearning
{
    [Serializable]
    public class Matrix : ICloneable, IEquatable<object> //Description of matrix operations from en.wikipedia.org/wiki/Matrix_(mathematics)
    {
        private int[] _dimensions;

        public int[] Dimensions
        {
            get
            {
                return this._dimensions;
            }
            set
            {
                foreach (int dimension in value)
                {
                    if (dimension < 0)
                    {
                        throw new ArgumentException("The dimensions for the matrix must be a positive, non-zero integers.", "Dimensions");
                    }
                }

                if (value.Length == 1)
                {
                    this._dimensions = new int[] { value[0], 1 };
                }
                else if (value.Length == 0)
                {
                    this._dimensions = new int[] { 1, 1 };
                }
                else
                {
                    this._dimensions = value.ToArray();
                }

                ResizeData();
            }
        }

        public double this[params int[] coords]
        {
            get
            { /* return the specified index here */
                int index = this.CoordinateToIndex(coords);
                return this.Data[index];
            }
            set
            { /* set the specified index to value here */
                int index = this.CoordinateToIndex(coords);
                this.Data[index] = value;
            }
        }

        public double this[int index]
        {
            get
            { /* return the specified index here */
                return this.Data[index];
            }
            set
            { /* set the specified index to value here */
                this.Data[index] = value;
            }
        }

        private double[] _data;
        private Array _cachedIndices;
        private int[][] _cachedCoordinates;

        public double[] Data
        {
            get
            {
                return this._data;
            }
            private set
            {
                this._data = value;
            }
        }

        public Matrix(params int[] dimensisons)
        {
            this.Dimensions = dimensisons;
        }

        public Matrix(IList<double> initialArray)
        {
            this.Dimensions = new int[] { initialArray.Count };
            Map((int[] coords) => initialArray[coords.First()]);
        }

        public void ResizeData()
        {
            int length = 1;
            for (int dimensionIndex = 0; dimensionIndex < this.Dimensions.Length; dimensionIndex++)
            {
                length *= this.Dimensions[dimensionIndex];
            }

            this._data = new double[length];
        }

        public Matrix Map(Func<double, double> func)
        {
            for (int index = 0; index < this.Data.Length; index++)
            {
                this.Data[index] = func(this.Data[index]);
            }

            return this;
        }

        public Matrix Map(Func<double, int[], double> func)
        {
            for (int index = 0; index < this.Data.Length; index++)
            {
                this.Data[index] = func(this.Data[index], this.IndexToCoordinate(index));
            }

            return this;
        }

        public Matrix Map(Func<double, int, double> func)
        {
            for (int index = 0; index < this.Data.Length; index++)
            {
                this.Data[index] = func(this.Data[index], index);
            }

            return this;
        }

        public Matrix Map(Func<int[], double> func)
        {
            for (int index = 0; index < this.Data.Length; index++)
            {
                this.Data[index] = func(this.IndexToCoordinate(index));
            }

            return this;
        }

        public Matrix Map(Func<int, double> func)
        {
            for (int index = 0; index < this.Data.Length; index++)
            {
                this.Data[index] = func(index);
            }

            return this;
        }

        public Matrix Map(Func<double> func)
        {
            for (int index = 0; index < this.Data.Length; index++)
            {
                this.Data[index] = func();
            }

            return this;
        }

        public Matrix AddScalar(double scalar)
        {
            Matrix newMatrix = this.Clone() as Matrix;

            newMatrix.Map((double data) =>
            {
                return data + scalar;
            });

            return newMatrix;
        }

        public static Matrix operator +(Matrix matrix, double scalar)
        {
            return matrix.AddScalar(scalar);
        }

        public static Matrix AddScalar(Matrix matrix, double scalar)
        {
            return matrix.AddScalar(scalar);
        }

        public Matrix SubtractScalar(double scalar)
        {
            Matrix newMatrix = this.Clone() as Matrix;

            newMatrix.Map((double data) =>
            {
                return data - scalar;
            });

            return newMatrix;
        }

        public static Matrix operator -(Matrix matrix, double scalar)
        {
            return matrix.SubtractScalar(scalar);
        }

        public static Matrix SubtractScalar(Matrix matrix, double scalar)
        {
            return matrix.SubtractScalar(scalar);
        }

        public Matrix MultiplyByScalar(double scalar)
        {
            Matrix newMatrix = this.Clone() as Matrix;

            newMatrix.Map((double data) =>
            {
                return data * scalar;
            });

            return newMatrix;
        }

        public static Matrix operator *(Matrix matrix, double scalar)
        {
            return matrix.MultiplyByScalar(scalar);
        }

        public static Matrix MultiplyByScalar(Matrix matrix, double scalar)
        {
            return matrix.MultiplyByScalar(scalar);
        }

        public Matrix DivideByScalar(double scalar)
        {
            Matrix newMatrix = this.Clone() as Matrix;

            newMatrix.Map((double data) =>
            {
                return data / scalar;
            });

            return newMatrix;
        }

        public static Matrix operator /(Matrix matrix, double scalar)
        {
            return matrix.DivideByScalar(scalar);
        }

        public static Matrix DivideByScalar(Matrix matrix, double scalar)
        {
            return matrix.DivideByScalar(scalar);
        }

        public Matrix MemberwiseAdd(Matrix toAdd)
        {
            if (!this.Dimensions.SequenceEqual(toAdd.Dimensions))
            {
                throw new ArgumentException("Both matrices must be of the same size for memberwise addition.", "toAdd");
            }

            Matrix newMatrix = this.Clone() as Matrix;

            newMatrix.Map((int index) =>
            {
                return newMatrix.Data[index] + toAdd.Data[index];
            });

            return newMatrix;
        }

        public static Matrix operator +(Matrix initialMatrix, Matrix toAdd)
        {
            return initialMatrix.MemberwiseAdd(toAdd);
        }

        public static Matrix MemberwiseAdd(Matrix initialMatrix, Matrix toAdd)
        {
            return initialMatrix.MemberwiseAdd(toAdd);
        }

        public Matrix MemberwiseSubtract(Matrix toSubtract)
        {
            if (!this.Dimensions.SequenceEqual(toSubtract.Dimensions))
            {
                throw new ArgumentException("Both matrices must be of the same size for memberwise subtraction.", "toSubtract");
            }

            Matrix newMatrix = this.Clone() as Matrix;

            newMatrix.Map((int index) =>
            {
                return newMatrix.Data[index] - toSubtract.Data[index];
            });

            return newMatrix;
        }

        public static Matrix operator -(Matrix initialMatrix, Matrix toSubtract)
        {
            return initialMatrix.MemberwiseSubtract(toSubtract);
        }

        public static Matrix MemberwiseSubtract(Matrix initialMatrix, Matrix toSubtract)
        {
            return initialMatrix.MemberwiseSubtract(toSubtract);
        }

        public Matrix MemberwiseMultiply(Matrix toMultiply)
        {
            if (!this.Dimensions.SequenceEqual(toMultiply.Dimensions))
            {
                throw new ArgumentException("Both matrices must be of the same size for memberwise multiplication.", "toMultiply");
            }

            Matrix newMatrix = this.Clone() as Matrix;

            newMatrix.Map((int index) =>
            {
                return newMatrix.Data[index] * toMultiply.Data[index];
            });

            return newMatrix;
        }

        public static Matrix operator ^(Matrix initialMatrix, Matrix toMultiply)
        {
            return initialMatrix.MemberwiseMultiply(toMultiply);
        }

        public static Matrix MemberwiseMultiply(Matrix initialMatrix, Matrix toMultiply)
        {
            return initialMatrix.MemberwiseMultiply(toMultiply);
        }

        public Matrix MemberwiseDivide(Matrix toDivide)
        {
            if (!this.Dimensions.SequenceEqual(toDivide.Dimensions))
            {
                throw new ArgumentException("Both matrices must be of the same size for memberwise division.", "toDivide");
            }

            Matrix newMatrix = this.Clone() as Matrix;

            newMatrix.Map((int index) =>
            {
                return newMatrix.Data[index] / toDivide.Data[index];
            });

            return newMatrix;
        }

        public static Matrix operator /(Matrix initialMatrix, Matrix toDivide)
        {
            return initialMatrix.MemberwiseDivide(toDivide);
        }

        public static Matrix MemberwiseDivide(Matrix initialMatrix, Matrix toDivide)
        {
            return initialMatrix.MemberwiseDivide(toDivide);
        }

        public static Tuple<int[], int[][], List<int>, List<int>> GenerateInfoForMatrixMultiplication(int[] leftMatrixDimensions, int[] rightMatrixDimensions)
        {
            List<int> newDimensions = new List<int>();
            List<int> commonDimensions = new List<int>();
            List<int> leftRemainder = new List<int>();
            List<int> rightRemainder = new List<int>();

            int[] leftDimensions = Utilities.TrimEndingElement(leftMatrixDimensions, 1).Reverse().ToArray();
            int[] rightDimensions = Utilities.TrimEndingElement(rightMatrixDimensions.Reverse().ToArray(), 1).Reverse().ToArray();

            int index = 0;
            do
            {
                if (leftDimensions[index] == rightDimensions[index])
                {
                    commonDimensions.Add(leftDimensions[index]);
                    index++;
                }
                else
                {
                    break;
                }
            } while (index < Math.Min(leftDimensions.Length, rightDimensions.Length) - 1);

            for (int leftRemainderIndex = index; leftRemainderIndex < leftDimensions.Length - 1; leftRemainderIndex++)
            {
                leftRemainder.Add(leftDimensions[leftRemainderIndex]);
            }
            for (int rightRemainderIndex = index; rightRemainderIndex < rightDimensions.Length - 1; rightRemainderIndex++)
            {
                rightRemainder.Add(rightDimensions[rightRemainderIndex]);
            }


            leftRemainder.Add(leftDimensions.Last());
            rightRemainder.Add(rightDimensions.Last());

            newDimensions.AddRange(leftRemainder.ToArray().Reverse());
            newDimensions.AddRange(rightRemainder.ToArray());

            int[][] commonIndexes = Utilities.GetPermutatedIndeces(commonDimensions.ToArray());

            return new Tuple<int[], int[][], List<int>, List<int>>(newDimensions.ToArray(), commonIndexes, leftRemainder, rightRemainder);
        }

        public Matrix MatrixMultiplication(Matrix rightMatrix)
        {
            if (this.Dimensions.Last() != rightMatrix.Dimensions.First())
            {
                throw new ArgumentException("The number of columns of the left matrix must be the same as the number of rows of the right matrix.", "rightMatrix");
            }

            Tuple<int[], int[][], List<int>, List<int>> tuple = Matrix.GenerateInfoForMatrixMultiplication(this.Dimensions, rightMatrix.Dimensions);
            Matrix newMatrix = new Matrix(tuple.Item1);
            int[][] commonIndexes = tuple.Item2;
            List<int> leftRemainder = tuple.Item3;
            List<int> rightRemainder = tuple.Item4;

            newMatrix.Map((int[] coords) =>
            {
                double sum = 0.0;

                foreach (int[] commonIndex in commonIndexes)
                {
                    int[] leftIndexes = coords.Take(leftRemainder.Count).Concat(commonIndex.Reverse()).ToArray();
                    double valLeft = this.Data[this.CoordinateToIndex(leftIndexes)];

                    int[] rightIndexes = commonIndex.Concat(coords.Skip(rightRemainder.Count)).ToArray();
                    double valRight = rightMatrix.Data[rightMatrix.CoordinateToIndex(rightIndexes)];

                    sum += valLeft * valRight;
                }

                return sum;
            });

            return newMatrix;
        }

        public static Matrix operator *(Matrix leftMatrix, Matrix rightMatrix)
        {
            return leftMatrix.MatrixMultiplication(rightMatrix);
        }
        public static Matrix MatrixMultiplication(Matrix leftMatrix, Matrix rightMatrix)
        {
            return leftMatrix.MatrixMultiplication(rightMatrix);
        }

        public Matrix Transpose()
        {
            int[] newDimenstions = this.Dimensions.Skip(1).Concat(this.Dimensions.Take(1)).ToArray();

            Matrix newMatrix = new Matrix(newDimenstions);

            for (int index = 0; index < this.Data.Length; index++)
            {
                int[] coords = this.IndexToCoordinate(index);
                int[] newIndex = coords.Skip(1).Concat(coords.Take(1)).ToArray();
                newMatrix[newIndex] = this.Data[index];
            }

            return newMatrix;
        }

        public static Matrix operator !(Matrix matrix)
        {
            return matrix.Transpose();
        }

        public static Matrix Transpose(Matrix matrix)
        {
            return matrix.Transpose();
        }

        public Matrix Flip()
        {
            return new Matrix(this.Dimensions).PopulateFromFlatData(this.GetFlatData().Reverse().ToArray());
        }

        public static Matrix Flip(Matrix matrix)
        {
            return matrix.Flip();
        }

        public static Matrix PrepareMatrixForFullConvolution(Matrix matrix, Matrix filter)
        {
            int[] newDimensions = new int[Math.Max(matrix.Dimensions.Length, filter.Dimensions.Length)];

            for (int dimensionIndex = 0; dimensionIndex < newDimensions.Length; dimensionIndex++)
            {
                int matrixDimension;
                try
                {
                    matrixDimension = matrix.Dimensions[dimensionIndex];
                }
                catch (Exception)
                {
                    matrixDimension = 1;
                }

                int filterDimension;
                try
                {
                    filterDimension = filter.Dimensions[dimensionIndex];
                }
                catch (Exception)
                {
                    filterDimension = 1;
                }

                newDimensions[dimensionIndex] = matrixDimension + (filterDimension - 1) * 2;
            }

            Matrix newMatrix = new Matrix(newDimensions);

            matrix.Map((double val, int[] coords) =>
            {
                int[] newCoords = new int[newDimensions.Length];

                for (int dimensionIndex = 0; dimensionIndex < newCoords.Length; dimensionIndex++)
                {
                    int matrixIndex, filterDimension;
                    try
                    {
                        matrixIndex = coords[dimensionIndex];
                    }
                    catch (Exception)
                    {
                        matrixIndex = 0;
                    }
                    try
                    {
                        filterDimension = filter.Dimensions[dimensionIndex];
                    }
                    catch (Exception)
                    {
                        filterDimension = 1;
                    }

                    newCoords[dimensionIndex] = matrixIndex + (filterDimension - 1);
                }

                newMatrix[newCoords] = val;

                return val;
            });

            return newMatrix;
        }

        public static int[] GenerateInfoForCrossCorrelation(int[] matrixDimensions, int[] filterDimensions)
        {
            List<int> newDimensions = new List<int>();

            for (int dimensionIndex = 0; dimensionIndex < matrixDimensions.Length; dimensionIndex++)
            {
                int operatedDimension, filterDimension;

                try
                {
                    operatedDimension = matrixDimensions[dimensionIndex];
                }
                catch (Exception)
                {
                    operatedDimension = 1;
                }
                try
                {
                    filterDimension = filterDimensions[dimensionIndex];
                }
                catch (Exception)
                {
                    filterDimension = 1;
                }

                int newDimension = operatedDimension - (filterDimension - 1);
                if (newDimension <= 0)
                {
                    throw new InvalidOperationException($"The filter is too big for the given matrix. Matrix dimension is {operatedDimension} and filter dimension is {filterDimension}.");
                }

                newDimensions.Add(newDimension);
            }

            return Matrix.SimplifyDimensions(newDimensions.ToArray());
        }

        public Matrix CrossCorrelation(Matrix filter, int[] newDimensions = null)
        {
            int thisRank = this.Dimensions.Length;
            int filterRank = filter.Dimensions.Length;
            if (thisRank < filterRank)
            {
                throw new ArgumentException($"The rank of the filter, {filterRank}, must be smaller or equal to the rank of the operated matrix, {thisRank}.", "filter");
            }

            Matrix newMatrix;

            if (newDimensions != null)
            {
                newMatrix = new Matrix(newDimensions);
            }
            else
            {
                newMatrix = new Matrix(Matrix.GenerateInfoForCrossCorrelation(this.Dimensions, filter.Dimensions));
            }

            int[] operatedCoords = new int[this.Dimensions.Length];

            newMatrix.Map((int[] newMatrixCoords) =>
            {
                double featureVal = 0;
                filter.Map((double filterVal, int[] filterCoords) =>
                {
                    //int[] operatedCoords = new int[this.Dimensions.Length];

                    for (int operatedIndex = 0; operatedIndex < operatedCoords.Length; operatedIndex++)
                    {
                        int newMatrixIndex, filterIndex;
                        try
                        {
                            newMatrixIndex = newMatrixCoords[operatedIndex];
                        }
                        catch (Exception)
                        {
                            newMatrixIndex = 0;
                        }
                        try
                        {
                            filterIndex = filterCoords[operatedIndex];
                        }
                        catch (Exception)
                        {
                            filterIndex = 0;
                        }

                        operatedCoords[operatedIndex] = newMatrixIndex + filterIndex;
                    }

                    featureVal += this[operatedCoords] * filterVal;

                    return filterVal;
                });

                return featureVal;
            });

            return newMatrix.Simplify();
        }

        public static Matrix CrossCorrelation(Matrix input, Matrix filter)
        {
            return input.CrossCorrelation(filter);
        }

        public Matrix Convolution(Matrix filter, int[] newDimensions = null)
        {
            return this.CrossCorrelation(filter.Flip(), newDimensions);
        }

        public static Matrix operator %(Matrix input, Matrix filter)
        {
            return input.Convolution(filter);
        }

        public static Matrix Convolution(Matrix input, Matrix filter)
        {
            return input.Convolution(filter);
        }

        public Tuple<Matrix, List<int[]>> MaxPool(Matrix filter, int[] newDimensions = null)
        {
            if (this.Data.Rank < filter.Data.Rank)
            {
                throw new ArgumentException("The rank of the filter must be smaller or equal to the rank of the operated matrix.", "filter");
            }

            Matrix newMatrix;

            if (newDimensions != null)
            {
                newMatrix = new Matrix(newDimensions);
            }
            else
            {
                newMatrix = new Matrix(Matrix.GenerateInfoForCrossCorrelation(this.Dimensions, filter.Dimensions));
            }
            List<int[]> maxValIndices = new List<int[]>();

            newMatrix.Map((int[] featureCoords) =>
            {
                double featureVal = 0;

                //Matrix featureCoordsMatrix = new Matrix(featureCoords.Length).Map((int[] coords) => featureCoords[coords[0]]);

                List<double> vals = new List<double>();
                List<int[]> indices = new List<int[]>();

                filter.Map((int[] filterCoords) =>
                {
                    //Matrix filterCoordsMatrix = new Matrix(filterCoords.Length).Map((int[] coords) => filterCoords[coords[0]]);
                    //Matrix operatedIndices = featureCoordsMatrix + filterCoordsMatrix;
                    //int[] index = operatedIndices.GetFlatData().Select(Convert.ToInt32).ToArray();

                    int[] operatedIndices = new int[featureCoords.Length];
                    for (int operatedIndex = 0; operatedIndex < operatedIndices.Length; operatedIndex++)
                    {
                        int featureIndex, filterIndex;
                        try
                        {
                            featureIndex = featureCoords[operatedIndex];
                        }
                        catch (Exception)
                        {
                            featureIndex = 0;
                        }
                        try
                        {
                            filterIndex = filterCoords[operatedIndex];
                        }
                        catch (Exception)
                        {
                            filterIndex = 0;
                        }

                        operatedIndices[operatedIndex] = featureIndex + filterIndex;
                    }

                    vals.Add(this[operatedIndices]);
                    indices.Add(operatedIndices);

                    return 0.0;
                });

                featureVal = vals.Max();
                maxValIndices.Add(indices[vals.IndexOf(featureVal)]);

                return featureVal;
            });

            return new Tuple<Matrix, List<int[]>>(newMatrix, maxValIndices);
        }

        public static Tuple<Matrix, List<int[]>> MaxPool(Matrix matrix, Matrix filter)
        {
            return matrix.MaxPool(filter);
        }

        public Matrix PopulateFromFlatData(params double[] data)
        {
            if (data.Length < this.Data.Length)
            {
                throw new ArgumentException("The lenght of provided data must be at least of the size of data of the matrix.", "data");
            }

            this.Data = data.ToArray();

            return this;
        }

        public double[] GetFlatData()
        {
            return this.Data.ToArray();
        }

        public double GetItem(params int[] indeces)
        {
            try
            {
                return this[indeces];
            }
            catch (Exception)
            {
                return 0.0;
            }
        }

        public static int[] SimplifyDimensions(int[] dimensions)
        {
            int[] newDimension = Utilities.TrimEndingElement(dimensions, 1).ToArray();
            newDimension = Utilities.TrimEndingElement(newDimension.Reverse().ToArray(), 1).Reverse().ToArray();

            return newDimension;
        }

        public Matrix Simplify()
        {
            Matrix simplifiedMatrix = new Matrix(Matrix.SimplifyDimensions(this.Dimensions)).PopulateFromFlatData(this.GetFlatData());

            return simplifiedMatrix;
        }

        public int[] IndexToCoordinate(int index)
        {
            int originalIndex = index;
            if (index > this.Data.Length - 1)
            {
                throw new IndexOutOfRangeException("Specified index is out of range for the matrix.");
            }

            if (this._cachedCoordinates == null)
            {
                this._cachedCoordinates = new int[this.Data.Length][];
            }

            int[] cachedCoordinate = this._cachedCoordinates[index];
            if (cachedCoordinate != null)
            {
                return cachedCoordinate;
            }
            else
            {
                int[] coordinate = new int[this.Dimensions.Length];

                int coefficient = 1;

                for (int dimensionIndex = 0; dimensionIndex < this.Dimensions.Length; dimensionIndex++)
                {
                    coefficient *= this.Dimensions[dimensionIndex];
                }

                for (int dimensionIndex = 0; dimensionIndex < this.Dimensions.Length; dimensionIndex++)
                {
                    coefficient /= this.Dimensions[dimensionIndex];
                    int dimensionCoordinate = index / coefficient;

                    index -= dimensionCoordinate * coefficient;

                    coordinate[dimensionIndex] = dimensionCoordinate;
                }

                this._cachedCoordinates[originalIndex] = coordinate;
                return coordinate;
            }
        }

        public static int[] IndexToCoordinate(int index, int[] dimensions)
        {
            int[] coordinate = new int[dimensions.Length];

            int coefficient = 1;

            for (int dimensionIndex = 0; dimensionIndex < dimensions.Length; dimensionIndex++)
            {
                coefficient *= dimensions[dimensionIndex];
            }

            for (int dimensionIndex = 0; dimensionIndex < dimensions.Length; dimensionIndex++)
            {
                coefficient /= dimensions[dimensionIndex];
                int dimensionCoordinate = index / coefficient;

                index -= dimensionCoordinate * coefficient;

                coordinate[dimensionIndex] = dimensionCoordinate;
            }

            return coordinate;
        }

        public int CoordinateToIndex(params int[] coordinate)
        {
            int thisRank = this.Dimensions.Length;
            int givenRank = coordinate.Length;
            if (thisRank != givenRank)
            {
                throw new ArgumentException($"Specified coordinates do not match the dimensions of the matrix ({thisRank} != {givenRank}).", "coordinate");
            }

            if (this._cachedIndices == null)
            {
                this._cachedIndices = Array.CreateInstance(typeof(int?), this.Dimensions);
            }

            int? cachedIndex = this._cachedIndices.GetValue(coordinate) as int?;
            if (cachedIndex.HasValue)
            {
                return cachedIndex.Value;
            }
            else
            {
                int index = 0;
                for (int dimensionIndex = 0; dimensionIndex < this.Dimensions.Length; dimensionIndex++)
                {
                    int higherDimensionCoefficient = 1;
                    for (int higherDimensionIndex = dimensionIndex + 1; higherDimensionIndex < this.Dimensions.Length; higherDimensionIndex++)
                    {
                        higherDimensionCoefficient *= this.Dimensions[higherDimensionIndex];
                    }
                    index += (higherDimensionCoefficient * coordinate[dimensionIndex]);
                }

                this._cachedIndices.SetValue(index, coordinate);

                return index;
            }
        }

        public static int CoordinateToIndex(int[] dimensions, params int[] coordinate)
        {
            if (coordinate.Length != dimensions.Length)
            {
                throw new ArgumentException("Specified coordinates do not match the dimensions of the matrix.", "coordinate");
            }

            int index = 0;
            for (int dimensionIndex = 0; dimensionIndex < dimensions.Length; dimensionIndex++)
            {
                int higherDimensionCoefficient = 1;
                for (int higherDimensionIndex = dimensionIndex + 1; higherDimensionIndex < dimensions.Length; higherDimensionIndex++)
                {
                    higherDimensionCoefficient *= dimensions[higherDimensionIndex];
                }
                index += (higherDimensionCoefficient * coordinate[dimensionIndex]);
            }

            return index;
        }

        public Array ToArray()
        {
            Array array = Array.CreateInstance(typeof(double), this.Dimensions);

            this.Map((double val, int[] coords) =>
            {
                array.SetValue(val, coords);
                return val;
            });

            return array;
        }

        public static Array ToArray(Matrix matrix)
        {
            return matrix.ToArray();
        }

        public int[] CoordinatesOf(double element)
        {
            return this.IndexToCoordinate(Array.IndexOf(this.Data, element));
        }

        public static int[] CoordinatesOf(Matrix matrix, double element)
        {
            return matrix.IndexToCoordinate(Array.IndexOf(matrix.Data, element));
        }

        public int IndexOf(double element)
        {
            return Array.IndexOf(this.Data, element);
        }

        public static int IndexOf(Matrix matrix, double element)
        {
            return Array.IndexOf(matrix.Data, element);
        }

        public override string ToString()
        {
            return $"Matrix dimension: {string.Join("-", this.Dimensions)}";
        }

        public object Clone()
        {
            return new Matrix(this.Dimensions)
            {
                Data = this.Data.ToArray()
            };
        }

        public static object Clone(Matrix matrix)
        {
            return new Matrix(matrix.Dimensions)
            {
                Data = matrix.Data.ToArray()
            };
        }

        // override object.Equals
        public override bool Equals(object obj)
        {
            if (obj == null || GetType() != obj.GetType())
            {
                return false;
            }
            else
            {
                Matrix toCompare = obj as Matrix;

                return this.Data.SequenceEqual(toCompare.Data) && this.Dimensions.SequenceEqual(toCompare.Dimensions);
            }
        }

        public static bool operator ==(Matrix left, Matrix right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Matrix left, Matrix right)
        {
            return !left.Equals(right);
        }

        // override object.GetHashCode
        public override int GetHashCode()
        {
            unchecked
            {
                const int HashingBase = 7655413;
                const int HashingMultiplier = 124693;

                int hash = HashingBase;
                hash = (hash * HashingMultiplier) ^ (!(this.Data is null) ? this.Data.GetHashCode() : 0);
                hash = (hash * HashingMultiplier) ^ (!(this.Dimensions is null) ? this.Dimensions.GetHashCode() : 0);

                return hash;
            }
        }
    }
}
