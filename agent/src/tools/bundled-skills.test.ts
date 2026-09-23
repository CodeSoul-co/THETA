import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, existsSync, readFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { BundledSkills } from './bundled-skills.js';
import type { ProductSession } from '../memory/session-store.js';

test('bundled Data Viz is source-pinned, readable and exports only selected templates',async()=>{
  const home=mkdtempSync(path.join(os.tmpdir(),'theta-skills-'));
  try{
    const skills=new BundledSkills(home),session:ProductSession={id:'skill-test',title:'Templates',datasetRefs:[],messages:[],updatedAt:''};
    const context={session,userMessage:'Prepare a figure template',save(){}};
    const catalog=await skills.execute('skills_list',{},context) as any;
    assert.match(catalog.skills[0].commit,/^[a-f0-9]{40}$/);assert.equal(catalog.skills[0].license,'MIT');
    const first=await skills.execute('skills_read',{skill:'data-viz',file:'SKILL.md',limit:100},context) as any;
    assert.equal(first.content.length,100);assert.equal(first.nextOffset,100);
    await assert.rejects(skills.execute('skills_read',{skill:'data-viz',file:'../../.env'},context),/text only|Unknown/);
    await assert.rejects(skills.execute('skills_prepare',{skill:'data-viz',figures:[999]},context),/Unknown/);
    const result=await skills.execute('skills_prepare',{skill:'data-viz',figures:[7]},context) as any;
    assert.equal(result.rendered,false);assert.ok(existsSync(path.join(result.workspace,'figures/figure07/plot.py')));
    assert.ok(!existsSync(path.join(result.workspace,'figures/figure01')));
    assert.match(readFileSync(path.join(result.workspace,'THETA-WORKSPACE.md'),'utf8'),/demonstrations, not user data/);
    const archive=execFileSync('tar',['-tzf',result.files[0].path],{encoding:'utf8'});
    assert.match(archive,/vizlib\/common.py/);assert.match(archive,/LICENSE/);assert.doesNotMatch(archive,/\.env|\.github/);
    assert.equal(session.skillArtifacts?.length,3);assert.equal(session.pendingConfirmation,undefined);
  }finally{rmSync(home,{recursive:true,force:true});}
});
