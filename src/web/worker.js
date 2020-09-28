onmessage = function(e) {
  const f = e.data[0];

  FS.mkdir('/working');
  FS.mount(WORKERFS, { files: [f] }, '/working');

  console.log('sketch result: ' + Module.sketch('/working/' + f.name, 15, 27, 2, 14, 156, false, true));
}

self.importScripts('web_sketch.js');