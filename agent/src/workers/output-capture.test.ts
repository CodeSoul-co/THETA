import assert from 'node:assert/strict';
import test from 'node:test';
import { OutputCapture } from './output-capture.js';
test('long output retains the final calculation with explicit omission metadata',()=>{
  const capture=new OutputCapture(1000);capture.push(Buffer.from('setup\n'+'x'.repeat(5000)+'\nFINAL VERIFIED RESULT=42'));
  const result=capture.finish();assert.equal(result.truncated,true);assert.ok(result.text.length<=1000);assert.match(result.text,/^setup/);assert.match(result.text,/OUTPUT TRUNCATED/);assert.match(result.text,/FINAL VERIFIED RESULT=42$/);assert.equal(result.bytes,5031);
});
test('UTF-8 split across chunks is preserved and short output is exact',()=>{
  const capture=new OutputCapture(1000),bytes=Buffer.from('中文 evidence');capture.push(bytes.subarray(0,1));capture.push(bytes.subarray(1,4));capture.push(bytes.subarray(4));
  assert.deepEqual(capture.finish(),{text:'中文 evidence',bytes:bytes.length,truncated:false});
});
