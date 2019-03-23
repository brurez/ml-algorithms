require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs-node");
const plot = require("node-remote-plot");

const loadCSV = require("../load-csv");
const LogisticRegression = require("./LogisticRegression");

const { features, labels, testFeatures, testLabels } = loadCSV(
  "./data/cars.csv",
  {
    dataColumns: ["horsepower", "displacement", "weight"],
    labelColumns: ["passedemissions"],
    shuffle: true,
    splitTest: 50,
    converters: {
      passedemissions: value => {
        return value === "TRUE" ? 1 : 0;
      }
    }
  }
);

const regression = new LogisticRegression(features, labels, {
  learningRate: 0.5,
  iterations: 5,
  batchSize: 50,
  decisionBoundary: 0.5,
});

regression.train();

console.log(regression.test(testFeatures, testLabels));

plot({
  x: regression.weightHistory,
  xLabel: "Iteration #",
  yLabel: "Weights",
  title: 'Logistic regression weights through iterations',
  name: 'logistic-regression-weights'
});

/*
plot({
  x: regression.costHistory.reverse(),
  xLabel: "Iteration #",
  yLabel: "Cost",
  title: 'Logistic regression costs through iterations',
  name: 'logistic-regression-cost'
});*/
