const tf = require("@tensorflow/tfjs");

class LinearRegressionTF {
  constructor(features, labels, options) {
    this.features = tf.tensor(features);
    this.labels = tf.tensor(labels);

    this.features = tf
      .ones([this.features.shape[0], 1])
      .concat(this.features, 1);

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

    this.weights = this.weights
      .sub(slopes.mul(this.options.learningRate));
  }

  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent();
    }
  }
}

module.exports = LinearRegressionTF;
