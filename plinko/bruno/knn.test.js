const knn = require('./knn');

describe("k Nearest Neighbor Algorithm", () => {
  const outputs = [
    [10, 0.5, 16, 1],
    [200, 0.5, 16, 4],
    [350, 0.5, 16, 4],
    [600, 0.5, 16, 5],
    [50, 0.5, 16, 1],
    [150, 0.5, 16, 4],
    [300, 0.5, 16, 4],
    [505, 0.5, 16, 5],
    [101, 0.5, 16, 1],
    [201, 0.5, 16, 3],
    [351, 0.5, 16, 4],
    [601, 0.5, 16, 5],
  ];

  it('a', () => {
      expect(knn(outputs, 300, 3)).toBe(4);
  });

  it('b', () => {
    expect(knn(outputs, 0, 5)).toBe(1);
  })
});
