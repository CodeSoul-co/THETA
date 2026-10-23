const { readFileSync } = require('node:fs');
const { createServer } = require('node:net');

function preferredPort(file, fallback = 14320) {
  try {
    const port = JSON.parse(readFileSync(file, 'utf8')).port;
    if (Number.isInteger(port) && port >= 0 && port <= 65535) return port;
  } catch { /* No saved port yet. */ }
  return fallback;
}

async function tryPort(port, serverFactory) {
  const server = serverFactory();
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(port, '127.0.0.1', resolve);
  });
  const selected = server.address().port;
  await new Promise((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
  return selected;
}

async function selectPort(preferred, serverFactory = createServer) {
  try { return await tryPort(preferred, serverFactory); }
  catch (error) {
    // Windows may reserve a port without any listening process, returning EACCES.
    if (preferred !== 0 && (error.code === 'EADDRINUSE' || error.code === 'EACCES')) {
      return tryPort(0, serverFactory);
    }
    throw error;
  }
}

module.exports = { preferredPort, selectPort };
