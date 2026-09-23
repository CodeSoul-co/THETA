import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { LocalProductTools } from './local-tools.js';
import { ProductSessionStore } from '../memory/session-store.js';
import { ResearchStore } from '../memory/research-store.js';
import type { ResearchRun, Dataset } from '../domain/research.js';
import { PythonCapabilityWorker } from '../adapters/python-worker.js';

test('real preprocessing binds a new version to the shared training editor without changing source or starting compute', async t => {
  const home = mkdtempSync(path.join(tmpdir(), 'theta-preprocessing-'));
  const sessions = new ProductSessionStore(home), records = new ResearchStore(home);
  const session = sessions.create(); const file = path.join(home, 'input.csv');
  const source = 'text,time\n  中文 English  ,2026-09-21\n第二篇研究,2026-09-20\n';
  writeFileSync(file, source);
  let submitted = 0;
  const tools = new LocalProductTools({runtimeDb:path.join(home,'runtime.sqlite'),uploadDir:path.join(home,'uploads'),worker:new PythonCapabilityWorker(),
    compute:{async submit(){submitted++;throw new Error('not authorized');},async status(){throw new Error('no jobs');},async cancel(){throw new Error('no jobs');},async results(){throw new Error('no jobs');}}});
  const context = {session,userMessage:'清洗正文并规范列名后配置 LDA，不执行',save:()=>sessions.save(session)};
  t.after(()=>{sessions.close();rmSync(home,{recursive:true,force:true});});
  await tools.attach(file,session); await tools.execute('run_create',{},context);
  const originalRun = session.runId!, originalRef = session.datasetRefs[0];
  const receipt = await tools.execute('dataset_preprocess',{purpose:'清理首尾空白并命名正文列',textColumn:'正文',timeColumn:'time',code:'df = df.rename(columns={"text": "正文"})\ndf["正文"] = df["正文"].str.strip()'},context) as {datasetRef:string};
  assert.notEqual(session.runId,originalRun);
  assert.equal(records.get<ResearchRun>('run',originalRun).datasetRef,originalRef);
  assert.equal(records.get<ResearchRun>('run',session.runId!).datasetRef,receipt.datasetRef);
  assert.equal(readFileSync(records.get<Dataset>('dataset',originalRef).managedPath,'utf8'),source);
  assert.equal(session.skillArtifacts?.length,3);
  assert.equal(sessions.get(session.id).datasetRefs.length,2);
  await tools.execute('training_configure',{plans:[{modelId:'lda',textColumn:'正文',params:{num_topics:2,max_iter:2},rationale:'仅小样本验收'}]},context);
  assert.equal(tools.trainingConfiguration(session)?.datasetRef,receipt.datasetRef);
  assert.equal(tools.trainingConfiguration(session)?.plans[0].textColumn,'正文');
  await assert.rejects(tools.execute('dataset_preprocess',{purpose:'不允许改已确认方案',textColumn:'正文',code:'df = df.copy()'},context),/待确认/);
  assert.equal(submitted,0);
});
