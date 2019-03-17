require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");
const LinearRegression = require("./LinearRegression");
const LinearRegressionTF = require("./LinearRegressionTF");

let { features, labels, testFeatures, testLabels } = loadCSV("./cars.csv", {
  shuffle: true,
  splitTest: 50,
  dataColumns: ["horsepower"],
  labelColumns: ["mpg"]
});

const regressionTF = new LinearRegressionTF(features, labels, {
  learningRate: 0.00001,
  iterations: 1000
});

regressionTF.train();
const R2 = regressionTF.test(testFeatures, testLabels);

console.log('R2 is: ', R2);

/*const regression = new LinearRegression(features, labels, {
  learningRate: 0.00002,
  iterations: 1000
});

regression.train();*/

//console.log("Updated M is: ", regression.m, "Updated B is: ", regression.b);
//console.log("Updated M is: ", regressionTF.weights.get(1, 0), "Updated B is: ", regressionTF.weights.get(0, 0));
