const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const tags = require('../pytorch-autotagger/data/tags.json');

const LIMIT = 50;
const PREPARED_IMAGE_SIZE = 224;
const MAX_COLOR_VALUE = 255;

async function getSortedTags(filepath) {
  let model = await tf.node.loadSavedModel(
    './model/danbooru/',
    ['serve'],
    'serving_default'
  );
  return tf.tidy(() => {
    let img = tf.image
      .resizeBilinear(tf.node.decodeImage(fs.readFileSync(filepath), 3), [
        PREPARED_IMAGE_SIZE,
        PREPARED_IMAGE_SIZE,
      ])
      .div(tf.scalar(MAX_COLOR_VALUE));
    img = tf.cast(img, (dtype = 'float32'));

    /*mean of natural image*/
    let meanRgb = { red: 0.485, green: 0.456, blue: 0.406 };

    /* standard deviation of natural image*/
    let stdRgb = { red: 0.229, green: 0.224, blue: 0.225 };

    let indices = [
      tf.tensor1d([0], 'int32'),
      tf.tensor1d([1], 'int32'),
      tf.tensor1d([2], 'int32'),
    ];

    /* sperating tensor channelwise and applyin normalization to each chanel seperately */
    let centeredRgb = {
      red: tf
        .gather(img, indices[0], 2)
        .sub(tf.scalar(meanRgb.red))
        .div(tf.scalar(stdRgb.red))
        .reshape([224, 224]),

      green: tf
        .gather(img, indices[1], 2)
        .sub(tf.scalar(meanRgb.green))
        .div(tf.scalar(stdRgb.green))
        .reshape([224, 224]),

      blue: tf
        .gather(img, indices[2], 2)
        .sub(tf.scalar(meanRgb.blue))
        .div(tf.scalar(stdRgb.blue))
        .reshape([224, 224]),
    };

    /* combining seperate normalized channels*/
    let processedImg = tf
      .stack([centeredRgb.blue, centeredRgb.green, centeredRgb.red])
      .expandDims();

    processedImg = processedImg
      .sub(tf.min(processedImg).dataSync()[0])
      .div(tf.max(processedImg).dataSync()[0] - tf.min(processedImg).dataSync()[0]);

    let scores = model.predict({ 'input.1': processedImg })['ret.11'];
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
  let filepaths = flattyReadDir('../dataset-for-tagging/original/');
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
