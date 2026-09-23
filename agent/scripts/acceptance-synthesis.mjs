import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import path from 'node:path';
import {loadThetaProjectEnvironment} from '../dist/src/environment.js';
import {createConfiguredProvider} from '../dist/src/providers/configured-provider.js';
import {LocalProductTools} from '../dist/src/tools/local-tools.js';
import {ProductSessionStore} from '../dist/src/memory/session-store.js';
import {ConversationAgent,observationText} from '../dist/src/conversation/conversation-agent.js';
if(!process.argv.includes('--confirm-synthesis')) throw new Error('Explicit confirmation required for this synthesis. No retraining occurs.');
loadThetaProjectEnvironment();
const home=path.resolve(process.argv[2]),training=JSON.parse(readFileSync(path.join(home,'training.json'),'utf8'));
const store=new ProductSessionStore(home),session=store.get(training.sessionId),save=()=>store.save(session);
const tools=new LocalProductTools({runtimeDb:path.join(home,'research.sqlite'),uploadDir:path.join(home,'uploads')});
const inference=createConfiguredProvider();assert.ok(inference);
try{
 const request=await tools.execute('results_synthesize',{jobIds:training.results.map(job=>job.id),question:'客服改善方向与 LDA/STM 差异：结合既有合成客服数据理解和这两份图表、表格逐项解释。STM原结果已发现均匀分布与主题退化；已定位为Python旧梯度恒零的代码缺陷，修复通过数值校验但未重跑，仅用于诊断；用户明确不重跑，不要建议增加迭代或再训练。LDA没有收敛诊断，不得声称收敛良好。不得用该STM结果推断真实渠道差异、不得给模型质量排名。LDA结果只支持合成样本的主题结构演示。'},{session,userMessage:'确认该次综合解读；不重跑，保留异常结果用于诊断',save});
 assert.ok(request.needsUser);
 const receipt=await tools.approve(session,'确认');save();
 const answer=await new ConversationAgent({inference,tools,save}).turn(session,'用户已明确确认本次综合解读。主机回执：'+observationText(receipt)+'。请直接完成已授权的具体问题，不提出或启动新训练。',undefined,{userIntent:''});
 const record=session.interpretations?.at(-1);assert.ok(record);
 writeFileSync(path.join(home,'synthesis.json'),JSON.stringify({documentPath:record.documentPath,answer,sourceJobs:training.results.map(job=>job.id)},null,2));
 console.log(JSON.stringify({documentPath:record.documentPath,answer:answer.slice(0,2200)}));
}finally{save();store.close();}
