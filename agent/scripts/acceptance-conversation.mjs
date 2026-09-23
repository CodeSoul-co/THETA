// Real configured conversational model, real dataset tools, strictly simulated compute gateway.
import assert from 'node:assert/strict';
import { writeSupportFixture } from './support-fixture.mjs';
import { mkdirSync, writeFileSync, readFileSync, copyFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { loadThetaProjectEnvironment } from '../dist/src/environment.js';
import { createConfiguredProvider } from '../dist/src/providers/configured-provider.js';
import { PythonCapabilityWorker, pythonExecutable } from '../dist/src/adapters/python-worker.js';
import { LocalProductTools } from '../dist/src/tools/local-tools.js';
import { ConversationAgent, observationText } from '../dist/src/conversation/conversation-agent.js';
import { ProductSessionStore } from '../dist/src/memory/session-store.js';
loadThetaProjectEnvironment();
const home = path.resolve(process.argv[2] ?? '.theta_agent/acceptance/conversation-' + Date.now());
mkdirSync(path.join(home, 'catalog'), { recursive:true });
const source = writeSupportFixture(path.join(home, 'outside-data', 'support feedback.csv'));
copyFileSync(source, path.join(home, 'catalog', 'support.csv'));
process.env.THETA_DATA_DIR = path.join(home, 'catalog');
// Never use the user's embedding credentials or actual compute endpoint in this fixture.
process.env.EMBEDDING_PROVIDER = 'cloud';
process.env.EMBEDDING_API_BASE = 'https://embedding.fixture.invalid/v1';
process.env.EMBEDDING_API_KEY = 'synthetic-only';
process.env.EMBEDDING_MODEL = 'fixture-embedding';
process.env.EMBEDDING_API_KEY_ENV = 'EMBEDDING_API_KEY';
const configured = createConfiguredProvider(); assert.ok(configured);
const inferenceLog=[];
const inference={id:configured.id,model:configured.model,async infer(request){const result=await configured.infer(request); inferenceLog.push({available:request.tools.map(t=>t.name),output:result.output});writeFileSync(path.join(home,'inference.json'),JSON.stringify(inferenceLog,null,2));return result;}};
const jobs = new Map(), reports = new Map(); let submissions = 0, reads = 0;
const compute = {
  async submit(request, receipt) {
    assert.ok(receipt); submissions++;
    const output = spawnSync(pythonExecutable(), ['scripts/fixture-report.py', home, request.jobId, request.plan.modelId], { encoding:'utf8' });
    assert.equal(output.status, 0, output.stderr);
    const fixture = JSON.parse(output.stdout);
    const job = { ...fixture.job, status:'completed', phase:'completed', percent:100 };
    jobs.set(job.id, job); reports.set(job.id, fixture.report); return job;
  },
  async status(id) { assert.ok(jobs.has(id)); return jobs.get(id); },
  async cancel() { throw new Error('No running fixture job'); },
  async results(id) { reads++; assert.ok(reports.has(id)); return reports.get(id); },
};
const worker = new PythonCapabilityWorker();
const tools = new LocalProductTools({ runtimeDb:path.join(home,'research.sqlite'), uploadDir:path.join(home,'uploads'), compute, worker });
const resuming = process.argv.includes('--resume');
const prior = resuming ? JSON.parse(readFileSync(path.join(home,'conversation.json'),'utf8')) : undefined;
const store = new ProductSessionStore(home); let session = prior ? store.get(prior.sessionId) : store.create();
if (prior) {
  submissions=prior.submissions; reads=prior.reads;
  for (const saved of session.reports ?? []) {
    const report=JSON.parse(readFileSync(saved.manifestPath,'utf8'));
    reports.set(report.jobId,report); jobs.set(report.jobId,{id:report.jobId,resultHash:report.resultHash,plan:{modelId:report.modelId},status:'completed',phase:'completed',percent:100});
  }
}
const transcript = prior?.transcript ?? []; const checks = (prior?.checks ?? []).filter(text=>!text.startsWith('FAILED:'));
function save() { store.save(session); }
async function turn(user, intent) {
  const start = Date.now(); const before = session.executionEvents?.length ?? 0;
  const agent = new ConversationAgent({ inference, tools, save, timeoutMs:150000 });
  const answer = await agent.turn(session, user, undefined, intent === undefined ? {} : {userIntent:intent});
  const events = (session.executionEvents ?? []).slice(before).filter(e=>e.status!=='started');
  transcript.push({ user, answer, elapsedMs:Date.now()-start, calls:events.map(e=>({name:e.name,status:e.status,durationMs:e.durationMs})), pending:session.pendingConfirmation?.summary });
  writeFileSync(path.join(home,'conversation.json'),JSON.stringify({home,sessionId:session.id,provider:inference.id,model:inference.model,submissions,reads,checks,transcript},null,2));
  console.log(JSON.stringify({turn:transcript.length,user:user.slice(0,60),answer:answer.slice(0,360),seconds:(Date.now()-start)/1000,tools:events.filter(e=>!e.name.includes('deepseek')).map(e=>e.name),pending:!!session.pendingConfirmation}));
  assert.ok(!answer.includes('本轮等待已达到时间上限'), 'Turn timed out');
  return answer;
}
async function pending(user, pattern) {
  await turn(user); assert.ok(session.pendingConfirmation, 'Expected reviewable host confirmation');
  assert.match(session.pendingConfirmation.summary,pattern);
}
async function confirm() {
  const outcome = await tools.approve(session,'确认'); save();
  return turn('模拟用户明确确认刚展示的操作。真实宿主回执（计算结果为模拟夹具）：'+observationText(outcome),'');
}
try {
  if (!resuming) {
  await turn('你好，主题模型可以帮我解决什么业务问题？');
  await turn('我数据已经传上去了，做一下文本分析。');
  assert.ok(transcript.at(-1).calls.some(e=>e.name==='datasets_discover'));
  checks.push('preuploaded discovery before understanding');
  await turn('使用 support.csv。我想找出客服体验中优先改进的问题，先理解内容，不要训练。');
  assert.equal(session.datasetRefs.length,1); assert.ok(session.contextId); checks.push('business understanding persisted');
  await turn('比较 LDA、STM 和需要云 embedding 的 THETA，给适合这些数据的模型和参数建议，目前只讨论。');
  assert.equal(submissions,0); checks.push('multi-model recommendations do not train');
  await pending('先用 LDA 做小实验：3个主题、最多5次迭代、英文图表、180秒、跳过评估和引擎图表。请展示本次训练确认。',/LDA|lda/);
  assert.equal(submissions,0); tools.deny(session); save();
  await pending('我拒绝刚才的 LDA。改用 STM，渠道 channel 作为协变量，其余限制不变，重新给我确认。',/stm|STM/);
  assert.equal(submissions,0); checks.push('denial and revised STM plan');
  await confirm(); assert.equal(submissions,1);
  if (!session.pendingConfirmation) await pending('请整理刚训练完成的结果，先给我确认。',/结果/);
  assert.equal(reads,0); tools.deny(session); save();
  await turn('这次先不解读结果。协变量是什么意思？'); assert.equal(reads,0); checks.push('result denial and side question');
  await pending('现在可以准备整理图表和表格了，先展示确认。',/结果/);
  await confirm(); assert.ok(reads>0); checks.push('approved report with full artifacts');
  await pending('结合之前的数据理解与这次图表，给我客服改善建议。请先确认本次综合解读。',/综合解读/);
  tools.deny(session); save();
  await pending('拒绝刚才过宽的解读。改成只讨论渠道差异能否支持改善决策，并说明因果局限，先给我确认。',/综合解读/);
  await confirm(); assert.ok(session.interpretations?.length); checks.push('revised synthesis saved separately');
  const savedId=session.id; session=store.get(savedId);
  await turn('继续刚才的会话，告诉我已经完成了哪些实验，别重新训练或解读。'); assert.equal(submissions,1); checks.push('history restored without repeated work');
  }
  await pending('另建研究尝试 THETA zero_shot，仍用此数据、3个主题，明确选择云 embedding，最多2次外部请求，180秒。直接生成正式训练授权卡，不要再问我是否要生成卡片；不执行训练。',/embedding|Embedding/);
  assert.match(session.pendingConfirmation.summary,/最多 2/); assert.equal(submissions,1); tools.deny(session);save(); checks.push('cloud plan bounded and denied without external calls');
  session=store.create();
  await turn('我有另外的文件，一会儿给你路径，只需要理解业务，不训练。');
  const attached=await tools.attach(source,session);save();
  await turn('用户只提供了文件。附件回执：'+observationText(attached)+'。没有新增指令，按前文继续；这不是训练授权。','');
  assert.ok(session.contextId); assert.equal(submissions,1); checks.push('outside-data file-only upload preserves intent');
  const contextId=session.contextId; session=store.create();
  await turn('我要复用之前的研究上下文，请列出已有的文档。');
  await turn('请选用这个上下文：'+contextId+'，概括研究目标，不要自动附加或训练。');
  assert.ok(session.contextId); assert.notEqual(session.contextId,contextId); assert.equal(session.datasetRefs.length,0); checks.push('new conversation reuses context without data/approval transfer');
} catch(error) {
  checks.push('FAILED: '+error.message); throw error;
} finally {
  writeFileSync(path.join(home,'conversation.json'),JSON.stringify({home,sessionId:session.id,provider:inference.id,model:inference.model,submissions,reads,checks,transcript},null,2));
  store.close(); console.log(JSON.stringify({home,checks}));
}
