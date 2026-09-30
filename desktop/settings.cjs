const path = require('node:path');
const root = require('node:fs').existsSync(path.join(__dirname, '../agent/runtime/settings.cjs'))
  ? path.join(__dirname, '../agent/runtime') : path.join(process.resourcesPath, 'runtime/agent/runtime');
module.exports = require(path.join(root, 'settings.cjs'));
