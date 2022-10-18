const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const tags = require('../pytorch-autotagger/data/tags.json');

const LIMIT = 50;
const PREPARED_IMAGE_SIZE = 224;
const MAX_COLOR_VALUE = 255;

const rgba2rgb = (rgba) => {
  if (rgba.shape[2] === 3) return rgba;
  if (rgba.shape[2] === 4)
    return tf.tidy(() => {
      const [r, g, b, a] = tf.unstack(rgba, 2);
      return tf.stack([r, g, b], 2);
    });
  throw new Error('invalid shape');
};

async function getSortedTags(filepath) {
  let model = await tf.node.loadSavedModel(
    './model/danbooru/',
    ['serve'],
    'serving_default'
  );
  return tf.tidy(() => {
    let tensor = rgba2rgb(
      tf.node.decodeImage(fs.readFileSync(filepath))
    ).resizeBilinear([PREPARED_IMAGE_SIZE, PREPARED_IMAGE_SIZE]);
    tensor = tensor.div(MAX_COLOR_VALUE);
    tensor = tensor.expandDims().transpose([0, 3, 1, 2]); // move color channel to 2nd place
    let scores = model.predict({ 'input.1': tensor })['ret.11'];
    let scoredTags = [];
    scores = scores.dataSync();
    for (let i in scores) {
      scoredTags.push({ score: scores[i], tag: tags[i] });
    }
    let sortedScoredTags = scoredTags.sort((a, b) => a.score - b.score);
    return sortedScoredTags
      .slice(sortedScoredTags.length - LIMIT, sortedScoredTags.length)
      .reverse();
  });
}

/**
 *
 * @param {string} dirPath
 * @returns {string[]}
 */
function flattyReadDir(dirPath) {
  return fs
    .readdirSync(dirPath, { withFileTypes: true })
    .filter((f) => f.isFile())
    .map((f) => path.join(dirPath, f.name));
}

(async () => {
  let result = {};
  let filepaths = flattyReadDir('../dataset-for-tagging/original2/');
  for (let i = 0; i < filepaths.length; i++) {
    let filepath = filepaths[i];
    result[filepath] = (await getSortedTags(filepath)).reduce(
      (accum, current) => {
        accum[current['tag']] = current['score'];
        return accum;
      },
      {}
    );
  }
  console.log(result);
})();
