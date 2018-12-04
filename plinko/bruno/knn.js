const { utils } = require('./utils');

const distance = (point1, point2) => {
  return Math.abs(point1 - point2);
};

const most = obj => {
  let topVal = 0;
  let topProp = null;
  for (let prop in obj) {
    if (obj[prop] > topVal) {
      topVal = obj[prop];
      topProp = prop;
    }
  }
  return topProp;
};

const knn = (outputs, target, k) => {
  const arr = outputs
    .map(row => [distance(row[0], target), row[3]])
    .sort((r1, r2) => r1[0] - r2[0])
    .slice(0, k);

  const commons =  arr.reduce((res, row) => {
      res[row[1]] ? res[row[1]]++ : (res[row[1]] = 1);
      return res;
    }, {});

  return Number.parseInt(most(commons));
};

module.exports = knn;
