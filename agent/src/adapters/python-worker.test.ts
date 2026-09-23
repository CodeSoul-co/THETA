import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,writeFileSync,rmSync} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {PythonCapabilityWorker} from './python-worker.js';

test('cancel kills a worker that ignores SIGTERM within the cleanup grace period',async()=>{
  const home=mkdtempSync(path.join(os.tmpdir(),'theta-worker-cancel-'));const previous=process.env.THETA_PYTHON;
  try{
    const executable=path.join(home,'ignore-term');
    writeFileSync(executable,'#!/usr/bin/env python3\nimport signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\nwhile True: time.sleep(0.05)\n',{mode:0o700});
    process.env.THETA_PYTHON=executable;const controller=new AbortController();const start=Date.now();
    const call=new PythonCapabilityWorker().call('test',{},controller.signal);const timer=setTimeout(()=>controller.abort(),500);
    await assert.rejects(call,/中断/);clearTimeout(timer);assert.ok(Date.now()-start<5000);
  }finally{if(previous===undefined)delete process.env.THETA_PYTHON;else process.env.THETA_PYTHON=previous;rmSync(home,{recursive:true,force:true});}
});
