import test from 'node:test';
import assert from 'node:assert/strict';
import { userMessageCount } from './message-count.js';
import type { ProductSession } from './session-store.js';

test('counts actual user sends, not assistant/tools/monitor prompts/confirmation clicks', () => {
  const session = { messages: [
    {role:'user',content:'分析这个数据'}, {role:'assistant',content:'好的'},
    {role:'tool',content:'receipt'}, {role:'user',content:'完成通知',metadata:{webHostEvent:true}},
    {role:'user',content:'确认',metadata:{webAction:true}},
    {role:'user',content:'确认训练配置：lda'},
    {role:'user',content:'确认训练配置：我手动输入的消息',metadata:{webUserText:'确认训练配置：我手动输入的消息'}},
  ] } as ProductSession;
  assert.equal(userMessageCount(session), 2);
  assert.equal(userMessageCount(JSON.parse(JSON.stringify(session))), 2);
});
