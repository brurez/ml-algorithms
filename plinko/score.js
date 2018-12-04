const output = [];

function sliceDataset(data, train = 50) {
  const s = _.shuffle(data);
  return [[...s.slice(0, train)], [...s.slice(train)]];
}

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  // Ran every time a balls drops into a bucket
  // console.log(arguments);
  output.push([...arguments]);
}

function runAnalysis() {
  const [train, test] = sliceDataset(output, 100);
  const buckets = [];

  for (let k = 1; k < 13; k++) {
    test.forEach(t => {
      buckets.push([knn(train, t[0], k), t[3]]);
    });
    console.log(
      "k = ",
      k,
      " - ",
      (countEqualPairs(buckets) / buckets.length).toFixed(2),
      "%"
    );
  }
}

function countEqualPairs(array) {
  return array.reduce((sum, item) => (item[0] === item[1] ? sum + 1 : sum), 0);
}
