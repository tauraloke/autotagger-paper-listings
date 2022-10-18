const fs = require('fs');

function hammingDistanceBetweenSets(list1, list2) {
  let count = 0;
  for (let i = 0; i < list1.length; i++) {
    if (!list2.includes(list1[i])) {
      count = count + 1;
    }
  }
  return count;
}

const hashSet1 = JSON.parse(
  fs.readFileSync(`../tagging-results/${process.argv[2]}.json`)
);
const hashSet2 = JSON.parse(
  fs.readFileSync(`../tagging-results/${process.argv[3]}.json`)
);
const filenames = Object.keys(hashSet1);
let total = 0;
for (let i in filenames) {
  let filename = filenames[i];
  if (!hashSet1[filename] || !hashSet2[filename]) {
    continue;
  }
  let distance =
    (100 / 50) *
    hammingDistanceBetweenSets(
      Object.keys(hashSet1[filename]),
      Object.keys(hashSet2[filename])
    );
  total = total + distance;
  console.log(distance + '%', filename);
}
console.log('----');
console.log(total / filenames.length + '%', 'total');
