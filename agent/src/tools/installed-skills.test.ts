import { Readable } from 'node:stream';
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { InstalledSkills } from './installed-skills.js';
import { BundledSkills } from './bundled-skills.js';
import type { ProductSession } from '../memory/session-store.js';

test('one persistent skill catalog serves UI and Agent, with safe import, enable, export and removal', async()=>{
  const home=mkdtempSync(path.join(tmpdir(),'theta-installed-skills-'));
  const files=[{path:'SKILL.md',content:Buffer.from('---\nname: research-demo\ndescription: Academic writing\n---\nUse evidence and write paragraphs.').toString('base64')},{path:'references/guide.md',content:Buffer.from('Reference example').toString('base64')}];
  try{
    const manager=new InstalledSkills(home), tools=new BundledSkills(home);
    const session:ProductSession={id:'skills',title:'Skills',datasetRefs:[],messages:[],updatedAt:''};
    const context={session,userMessage:'Use my imported skill',save(){}};
    assert.equal(manager.import(files).id,'research-demo');
    assert.equal(manager.list().length,2);
    assert.match((await tools.execute('skills_read',{skill:'research-demo'},context) as any).content,/write paragraphs/);
    manager.setEnabled('research-demo',false);
    await assert.rejects(tools.execute('skills_read',{skill:'research-demo'},context),/disabled/);
    assert.equal((await tools.execute('skills_list',{},context) as any).skills.length,1);
    manager.setEnabled('research-demo',true);
    const archive=await manager.archive('research-demo'); assert.ok(archive.length>50);
    manager.remove('research-demo');await manager.importArchive(Readable.from([archive]));
    assert.equal(manager.list().length,2);
    assert.throws(()=>manager.import([...files,{path:'../outside.txt',content:'YQ=='}], 'local','bad-path'),/Unsafe/);
    assert.throws(()=>manager.import([...files,{path:'SKILL.MD',content:'YQ=='}], 'local','duplicate'),/Duplicate/);
    assert.throws(()=>manager.import(files,'local','data-viz'),/Invalid/);
    assert.throws(()=>manager.remove('../data-viz'),/Invalid/);
    await assert.rejects(manager.download('http://127.0.0.1/secret'),/GitHub/);
    manager.remove('research-demo');assert.equal(manager.list().length,1);
  }finally{rmSync(home,{recursive:true,force:true});}
});
