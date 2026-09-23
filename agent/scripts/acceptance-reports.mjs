import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { loadThetaProjectEnvironment } from '../dist/src/environment.js';
import { createConfiguredProvider } from '../dist/src/providers/configured-provider.js';
import { LocalProductTools } from '../dist/src/tools/local-tools.js';
import { ProductSessionStore } from '../dist/src/memory/session-store.js';
import { ConversationAgent, observationText } from '../dist/src/conversation/conversation-agent.js';
if (!process.argv.includes('--confirm-reports')) throw new Error('Explicit confirmation required to read the two trained results and generate/interpret reports.');
loadThetaProjectEnvironment();
const home=path.resolve(process.argv[2]);
const training=JSON.parse(readFileSync(path.join(home,'training.json'),'utf8'));
const store=new ProductSessionStore(home), session=store.get(training.sessionId);
const tools=new LocalProductTools({runtimeDb:path.join(home,'research.sqlite'),uploadDir:path.join(home,'uploads')});
const save=()=>store.save(session);
const inference=createConfiguredProvider(); assert.ok(inference);
const reports=[];
try {
  for(const job of training.results) {
    const context={session,userMessage:'确认生成并解读两份结果报告',save};
    await tools.execute('run_select',{runId:job.runId},context);
    const request=await tools.execute('results_read',{view:'report',jobId:job.id},context);
    const receipt=request.needsUser ? await tools.approve(session,'确认') : {resumedAction:'results.read',reports:[request]};save();
    const answer=await new ConversationAgent({inference,tools,save}).turn(session,'用户已确认读取并基础解读两份真实本地训练结果。主机回执：'+observationText(receipt)+'。输入是36条合成客服文本。请逐一说明图表、表格与缺失指标；这不授权业务综合解读，不要另行发起它。',undefined,{userIntent:''});
    const report=receipt.reports[0];
    writeFileSync(path.join(path.dirname(report.reportPath),'basic-interpretation.md'),answer);
    reports.push({...report,basicInterpretationPath:path.join(path.dirname(report.reportPath),'basic-interpretation.md')});
    writeFileSync(path.join(home,'reports.json'),JSON.stringify(reports,null,2));
    console.log(JSON.stringify({model:job.modelId,reportPath:report.reportPath,files:report.files.map(f=>f.name),missing:report.missingEvidence,summary:report.summary}));
  }
} finally {save();store.close();}
