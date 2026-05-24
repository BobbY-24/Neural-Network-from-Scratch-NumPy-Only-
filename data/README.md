# Data

I use the Kaggle Digit Recognizer / MNIST-style training CSV for this project.

## Expected File

Place the extracted CSV here:

```text
data/train.csv
```

The expected format is:

- first column: digit label from `0` to `9`,
- remaining columns: flattened 28x28 grayscale pixel values.

## Why The Raw Zip Is Not Tracked

I removed the raw downloaded zip from Git tracking because it is a large dataset artifact, not source code. Keeping the repository lightweight makes it easier to clone and review.

## How To Prepare Data

1. Download the Digit Recognizer dataset from Kaggle.
2. Extract `train.csv`.
3. Place it at `data/train.csv`.

The training command expects that path by default.
