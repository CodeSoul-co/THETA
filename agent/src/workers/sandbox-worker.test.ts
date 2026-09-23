import assert from 'node:assert/strict';
import test from 'node:test';
import { DockerSandboxWorker } from './sandbox-worker.js';

const image='sha256:'+'a'.repeat(64);
test('worker identity tracks image and resource configuration; description is not mutable',()=>{
  const worker=new DockerSandboxWorker({image});
  const original=worker.describe();assert.equal(original.network,'none');assert.equal(original.persistentVariables,false);
  assert.equal(original.fingerprint,new DockerSandboxWorker({image}).describe().fingerprint);
  for(const options of [{image,cpus:2},{image,memoryMiB:2048},{image,maxSeconds:30},{image:'sha256:'+'b'.repeat(64)}])assert.notEqual(original.fingerprint,new DockerSandboxWorker(options).describe().fingerprint);
  original.cpus=20;assert.equal(worker.describe().cpus,4);
});
test('worker refuses unpinned images and invalid resource limits',()=>{
  for(const options of [{image:'latest'},{image,cpus:0},{image,memoryMiB:Infinity},{image,maxSeconds:NaN},{image,maxSeconds:200}])assert.throws(()=>new DockerSandboxWorker(options),/Invalid bounded/);
});
