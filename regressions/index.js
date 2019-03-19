require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const plot = require("node-remote-plot");
const loadCSV = require("./load-csv");
// const LinearRegression = require("./LinearRegression");
const LinearRegressionTF = require("./LinearRegressionTF");

let { features, labels, testFeatures, testLabels } = loadCSV("./cars.csv", {
  shuffle: true,
  splitTest: 50,
  dataColumns: ["horsepower", "weight", "displacement"],
  labelColumns: ["mpg"]
});

const regressionTF = new LinearRegressionTF(features, labels, {
  learningRate: 0.1,
  iterations: 20
});

regressionTF.features.print();
regressionTF.labels.print();

regressionTF.train();
const r2 = regressionTF.test(testFeatures, testLabels);

plot({
  x: regressionTF.mseHistory.reverse(),
  xLabel: 'Iteration #',
  yLabel: 'Mean squared error'
});

console.log("R2 is: ", r2);

/*const regression = new LinearRegression(features, labels, {
  learningRate: 0.00002,
  iterations: 1000
});

regression.train();*/

//console.log("Updated M is: ", regression.m, "Updated B is: ", regression.b);
//console.log("Updated M is: ", regressionTF.weights.get(1, 0), "Updated B is: ", regressionTF.weights.get(0, 0));
