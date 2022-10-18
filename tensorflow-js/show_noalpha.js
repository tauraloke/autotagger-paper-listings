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
    let tensor = tf.node
      .decodeImage(fs.readFileSync(filepath), 3)
      .resizeBilinear([PREPARED_IMAGE_SIZE, PREPARED_IMAGE_SIZE]);
    tf.node.encodeJpeg(tensor, 'rgb').then((image) => {
      fs.writeFileSync(
        filepath.replace('/original/', '/tfjs-tidy/224_'),
        Buffer.from(image)
      );
    });

    return [];
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
