const { randomBytes, createCipheriv, createDecipheriv } = require('node:crypto');
const { readFileSync, writeFileSync, chmodSync } = require('node:fs');
const path = require('node:path');

// User-owned encryption avoids Keychain access dialogs after unsigned upgrades.
function createCredentials(home, legacyDecrypt) {
  const file = path.join(home, 'credentials.key');
  function key() {
    try { writeFileSync(file, randomBytes(32), { flag: 'wx', mode: 0o600 }); }
    catch (error) { if (error.code !== 'EEXIST') throw error; }
    if (process.platform !== 'win32') chmodSync(file, 0o600);
    const bytes = readFileSync(file);
    if (bytes.length !== 32) throw new Error('API Key 加密文件损坏，请重新配置');
    return bytes;
  }
  return {
    encrypt(value) {
      const iv = randomBytes(12), cipher = createCipheriv('aes-256-gcm', key(), iv);
      const data = Buffer.concat([cipher.update(value, 'utf8'), cipher.final()]);
      return 'local:v1:' + Buffer.concat([iv, cipher.getAuthTag(), data]).toString('base64');
    },
    decrypt(value) {
      if (!value.startsWith('local:v1:')) { if (legacyDecrypt) return legacyDecrypt(value); throw new Error('旧版 API Key 请重新输入一次'); }
      const bytes = Buffer.from(value.slice(9), 'base64');
      const decipher = createDecipheriv('aes-256-gcm', key(), bytes.subarray(0, 12));
      decipher.setAuthTag(bytes.subarray(12, 28));
      return Buffer.concat([decipher.update(bytes.subarray(28)), decipher.final()]).toString('utf8');
    },
  };
}
module.exports = { createCredentials };
