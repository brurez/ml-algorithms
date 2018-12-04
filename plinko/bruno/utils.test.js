
const utils = require('./utils');

describe('utils', () => {
  it('#minMax', () => {
    const test = [
      [0, 1],
      [2, 2],
      [10, 3]
    ];

    expect(utils.minMax(test, 2)).toEqual([[0, 0], [0.2, 0.5], [1, 1]]);
  })
});
