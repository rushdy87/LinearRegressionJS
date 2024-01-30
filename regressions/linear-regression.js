const tf = require('@tensorflow/tfjs');

class LinearRegression {
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
    const currentGuesses = this.features.matMul(this.weights);
    const differences = currentGuesses.sub(this.labels);

    const slopes = this.features
      .transpose()
      .matMul(differences)
      .div(this.features.shape[0]);

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

    const prodictions = testFeatures.matMul(this.weights);

    const res = testLabels.sub(prodictions).pow(2).sum().arraySync();

    const tot = testLabels.sub(testLabels.mean()).pow(2).sum().arraySync();

    return 1 - res / tot;
  }

  processFeatures(features) {
    features = tf.tensor(features);
    features = tf.ones([features.shape[0], 1]).concat(features, 1);
    return features;
  }
}

module.exports = LinearRegression;

/*
This is the old version:
gradientDescent() {
    const currentGuessesForMPG = this.features.map((row) => {
      return this.m * row[0] + this.b;
    });

    const bSlope =
      (_.sum(
        currentGuessesForMPG.map((guess, i) => guess - this.labels[i][0])
      ) *
        2) /
      this.features.length;

    const mSlope =
      (_.sum(
        currentGuessesForMPG.map(
          (guess, i) => -1 * this.features[i][0] * (this.labels[i][0] - guess)
        )
      ) *
        2) /
      this.features.length;

    this.m = this.m - mSlope * this.options.learningRate;
    this.b = this.b - bSlope * this.options.learningRate;
  }
*/
