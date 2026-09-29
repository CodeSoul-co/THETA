import { createWriteStream } from 'node:fs';
import { rm } from 'node:fs/promises';
import type { IncomingMessage } from 'node:http';
import { Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';

/** Backpressure keeps upload memory bounded; file length has no product cap. */
export async function receiveUpload(request: IncomingMessage, destination: string): Promise<number> {
  let bytes = 0;
  const counter = new Transform({ transform(chunk, _encoding, callback) { bytes += chunk.length; callback(null, chunk); } });
  try {
    await pipeline(request, counter, createWriteStream(destination, { flags: 'wx', mode: 0o600 }));
    if (!bytes) throw new Error('不能上传空文件');
    const expected = request.headers['x-theta-file-size'];
    if (expected !== undefined && (!/^\d+$/.test(String(expected)) || Number(expected) !== bytes)) throw new Error('文件传输不完整，请重新上传原始文件。');
    return bytes;
  } catch (error) {
    await rm(destination, { force: true });
    throw error;
  }
}
