const tf = require("@tensorflow/tfjs");
require("@tensorflow/tfjs-node");
const loadCSV = require("./load-csv");

function knn(features, labels, predition, k) {
  const { mean, variance } = tf.moments(features, 0);

  const scaledPrediction = predition
    .sub(mean)
    .div(variance.pow(0.5));

  return (
    features
      .sub(mean)
      .div(variance.pow(0.5))
      .sub(scaledPrediction)
      .pow(2)
      .sum(1)
      .pow(0.5)
      .expandDims(1)
      .concat(labels, 1)
      .unstack()
      .sort((t1, t2) => (t1.get(0) > t2.get(0) ? 1 : -1))
      .slice(0, k)
      .reduce((acc, t) => t.get(1) + acc, 0) / k
  );
}

let { features, labels, testFeatures, testLabels } = loadCSV(
  "kc_house_data.csv",
  {
    shuffle: true,
    splitTest: 10,
    dataColumns: ["lat", "long", "sqft_lot", 'sqft_living'],
    labelColumns: ["price"]
  }
);

features = tf.tensor(features);
labels = tf.tensor(labels);

testFeatures.forEach((test, i) => {
  const result = knn(features, labels, tf.tensor(test), 10);
  const err = (testLabels[i][0] - result) / testLabels[i][0];

  console.log('Guess', result, testLabels[i][0]);
  console.log("Error", err * 100);
});
