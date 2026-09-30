const path = require('node:path');
const root = require('node:fs').existsSync(path.join(__dirname, '../agent/runtime/credentials.cjs'))
  ? path.join(__dirname, '../agent/runtime') : path.join(process.resourcesPath, 'runtime/agent/runtime');
module.exports.createCredentials = require(path.join(root, 'credentials.cjs')).createCredentials;
