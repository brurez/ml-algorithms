const tf = require("@tensorflow/tfjs");

class LogisticRegression {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    this.costHistory = [];
    this.weightHistory = [];

    this.options = Object.assign(
      {
        learningRate: 0.1,
        iterations: 100,
        batchSize: 10,
        decisionBoundary: 0.5
      },
      options
    );

    const wRows = this.features.shape[1];

    this.weights = tf.zeros([wRows, 1]);
  }

  gradientDescent(features, labels) {
    const mxPlusBs = features.matMul(this.weights).sigmoid();
    const differences = mxPlusBs.sub(labels);
    const n = features.shape[0];

    const slopes = features
      .transpose()
      .matMul(differences)
      .div(n);

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
    this.weightHistory.push(Array.from(this.weights.dataSync()));
  }

  train() {
    const { batchSize } = this.options;
    const batchQuantity = Math.floor(this.features.shape[0] / batchSize);

    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const featSlice = this.features.slice(
          [j * batchSize, 0],
          [batchSize, -1]
        );
        const labelSlice = this.labels.slice(
          [j * batchSize, 0],
          [batchSize, -1]
        );

        this.gradientDescent(featSlice, labelSlice);
      }

      this.recordCost();
      this.updateLearningRate();
    }
  }

  predict(observations) {
    return this.processFeatures(observations)
      .matMul(this.weights)
      .sigmoid()
      .greater(this.options.decisionBoundary)
      .cast("float32");
  }

  test(testFeatures, testLabels) {
    const predictions = this.predict(testFeatures);
    testLabels = tf.tensor(testLabels);

    const incorrect = predictions
      .sub(testLabels)
      .abs()
      .sum()
      .get();

    return (predictions.shape[0] - incorrect) / predictions.shape[0];
  }

  processFeatures(features) {
    features = tf.tensor(features);
    features = this.standardize(features);
    features = tf.ones([features.shape[0], 1]).concat(features, 1);

    return features;
  }

  standardize(features) {
    if (!this.mean || !this.variance) {
      const { mean, variance } = tf.moments(features, 0);
      this.mean = mean;
      this.variance = variance;
    }

    return features.sub(this.mean).div(this.variance.pow(0.5));
  }

  recordCost() {
    const guesses = this.features.matMul(this.weights).sigmoid();

    const termOne = this.labels.transpose().matMul(guesses.log());

    const termTwo = this.labels
      .mul(-1)
      .add(1)
      .transpose()
      .matMul(
        guesses
          .mul(-1)
          .add(1)
          .log()
      );

    const cost = termOne
      .add(termTwo)
      .div(this.features.shape[0])
      .mul(-1)
      .get(0, 0);

    this.costHistory.unshift(cost);
  }

  updateLearningRate() {
    if (this.costHistory.length < 2) return;
    const [last, secondLast] = this.costHistory;
    if (last > secondLast) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = LogisticRegression;
