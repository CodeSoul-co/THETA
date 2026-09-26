const fs = require('node:fs');
const path = require('node:path');
const { createHash } = require('node:crypto');
const yaml = require('js-yaml');
const { version } = require('../package.json');
(async () => {
  const directory = path.resolve(__dirname, '../release');
  const name = process.platform === 'darwin' ? 'latest-mac.yml' : 'latest.yml';
  const info = yaml.load(fs.readFileSync(path.join(directory, name), 'utf8'));
  if (info.version !== version || !info.files?.length) throw Error('Missing current update metadata');
  for (const file of info.files) {
    if (path.basename(file.url) !== file.url || !file.url.startsWith(`THETA-${version}-`)) throw Error('Update metadata must reference local release assets');
    const hash = createHash('sha512');
    let bytes = 0;
    for await (const chunk of fs.createReadStream(path.join(directory, file.url))) { hash.update(chunk); bytes += chunk.length; }
    if (hash.digest('base64') !== file.sha512 || bytes !== file.size) throw Error(`Update checksum mismatch: ${file.url}`);
  }
  if (process.platform === 'darwin' && !info.files.some(file => file.url.endsWith('.zip'))) throw Error('Native Mac updates require ZIP');
  if (process.platform === 'win32' && !info.files.some(file => file.url.endsWith('.exe'))) throw Error('Windows updates require EXE');
  console.log(`Verified ${name}: version ${version}, ${info.files.length} assets and SHA-512 checksums`);
})().catch(error => { console.error(error); process.exitCode = 1; });
