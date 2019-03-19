const tf = require("@tensorflow/tfjs");

class LinearRegressionTF {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    this.mseHistory = [];

    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000, batchSize: 10 },
      options
    );

    const wRows = this.features.shape[1];

    this.weights = tf.zeros([wRows, 1]);
  }

  gradientDescent(features, labels) {
    const mxPlusBs = features.matMul(this.weights);
    const differences = mxPlusBs.sub(labels);
    const n = features.shape[0];

    const slopes = features
      .transpose()
      .matMul(differences)
      .div(n);

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
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

      this.recordMSE();
      this.updateLearningRate();
    }
  }

  predict(observations) {
    return this.processFeatures(observations).matMul(this.weights);
  }

  test(testFeatures, testLabels) {
    testFeatures = this.processFeatures(testFeatures);
    testLabels = tf.tensor(testLabels);

    const predictions = testFeatures.matMul(this.weights);

    const SSres = testLabels
      .sub(predictions)
      .pow(2)
      .sum()
      .get();

    const SStot = testLabels
      .sub(testLabels.mean())
      .pow(2)
      .sum()
      .get();

    // Coeffiecient of Determination
    const r2 = 1 - SSres / SStot;

    return r2;
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

  recordMSE() {
    const mse = this.features
      .matMul(this.weights)
      .sub(this.labels)
      .pow(2)
      .sum()
      .div(this.features.shape[0])
      .get();
    this.mseHistory.unshift(mse);
  }

  updateLearningRate() {
    if (this.mseHistory.length < 2) return;
    const [last, secondLast] = this.mseHistory;
    if (last > secondLast) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = LinearRegressionTF;
