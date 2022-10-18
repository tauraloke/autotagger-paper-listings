const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
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
  let sharpedImage = sharp(filepath).removeAlpha();
  sharpedImage = sharpedImage.resize(PREPARED_IMAGE_SIZE, PREPARED_IMAGE_SIZE, {
    fit: sharp.fit.cover,
    kernel: sharp.kernel.mitchell,
  });
  let buffer = await sharpedImage.toBuffer();
  return tf.tidy(() => {
    let tensor = tf.node.decodeImage(buffer, 3);
    //tf.node.encodeJpeg(tensor, 'rgb').then((image) => {
    //  fs.writeFileSync(
    //    filepath.replace('/original2/', '/tfjs-sharp-no-opacity/224_'),
    //    Buffer.from(image)
    //  );
    //});
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
