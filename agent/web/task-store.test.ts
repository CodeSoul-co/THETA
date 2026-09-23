import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,rmSync} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {ResearchStore} from '../src/memory/research-store.js';
import {ProductSessionStore} from '../src/memory/session-store.js';
import {WebTaskStore,publicTask} from './task-store.js';

test('dead owner recovery retains the task without replay and releases only its own lease',()=>{
  const home=mkdtempSync(path.join(os.tmpdir(),'theta-task-recovery-'));
  const sessions=new ProductSessionStore(home);
  try {
    const s=sessions.create();const lease=sessions.acquire(s.id);
    const store=new WebTaskStore(new ResearchStore(home));
    const task=store.create(s.id,'request-one',{content:'Analyze'},'Analyze');
    store.update(task.id,{status:'running',ownerPid:2147483647,lease});
    // Recovery must also reach old tasks after more than one page of newer receipts.
    for (let n=0;n<105;n++) store.create(s.id,`newer-${n}`,{},'Newer task');
    store.recover((id,token)=>sessions.release(id,token));
    const recovered=store.get(task.id);assert.equal(recovered.status,'interrupted');
    assert.deepEqual(recovered.request,{content:'Analyze'});
    const next=sessions.acquire(s.id);sessions.release(s.id,next);
    assert.equal(store.find(s.id,'request-one')?.id,task.id);
    assert.ok(!('lease' in publicTask(recovered)));assert.ok(!('request' in publicTask(recovered)));
  } finally {sessions.close();rmSync(home,{recursive:true,force:true});}
});
