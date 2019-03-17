const tf = require("@tensorflow/tfjs");

class LinearRegressionTF {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);

    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000 },
      options
    );

    this.weights = tf.zeros([2, 1]);
  }

  gradientDescent() {
    const mxPlusBs = this.features.matMul(this.weights);
    const differences = mxPlusBs.sub(this.labels);
    const n = this.features.shape[0];

    const slopes = this.features
      .transpose()
      .matMul(differences)
      .div(n);

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent();
    }
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
    const R2 = 1 - SSres / SStot;

    return R2;
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

    return features
      .sub(this.mean)
      .div(this.variance.pow(0.5));
  }
}

module.exports = LinearRegressionTF;
