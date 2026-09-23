import { createHash, randomUUID } from 'node:crypto';
import { existsSync, lstatSync, mkdirSync, readFileSync, readdirSync, realpathSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import type { ProductSession } from '../memory/session-store.js';
import { DockerSandboxWorker, type SandboxWorker, type SandboxDescription } from '../workers/sandbox-worker.js';

/** A binding is a HOST grant, never arguments supplied by the model. Bind only
 * explicitly authorized data/results and an isolated workspace to this study. */
export interface AnalysisBinding {
  scope: string; workspace: string; inputs: string; model: string;
  image: string; maxExecutions: number | null; maxSeconds: number;
  workerFingerprint?: string;
}
export interface AnalysisRuntime {
  inspect(session: ProductSession): unknown;
  execute(session: ProductSession, request: AnalysisExecution, signal?: AbortSignal): Promise<unknown>;
  deliver(session: ProductSession, files: string[]): unknown;
}
export interface AnalysisExecution { purpose: string; code: string; outputs: string[]; seconds: number }
interface FileReceipt { path: string; bytes: number; sha256: string }
interface RunReceipt { id: string; purpose: string; script: FileReceipt; exitCode: number | null; outputs: FileReceipt[]; changedOutputs: string[]; missing: string[]; seconds: number; error?: string; worker?: SandboxDescription; cleanupConfirmed?: boolean }
interface Ledger { executions: RunReceipt[]; delivered?: FileReceipt[] }
const hash = (value: string | Buffer) => createHash('sha256').update(value).digest('hex');

export function workspaceFile(root: string, relative: string): string {
  if (!relative || relative.includes('\\') || path.isAbsolute(relative) || relative.split('/').some(p => !p || p === '.' || p === '..')) throw new Error('Expected a relative workspace file path');
  if (lstatSync(root).isSymbolicLink()) throw new Error('Symlink workspace rejected');
  let file = root;
  for (const part of relative.split('/')) { file = path.join(file, part); if (existsSync(file) && lstatSync(file).isSymbolicLink()) throw new Error('Symlink artifact rejected'); }
  return file;
}
function receipt(root: string, name: string): FileReceipt {
  const file = workspaceFile(root, name), stat = lstatSync(file);
  if (!stat.isFile() || stat.nlink !== 1 || stat.size === 0 || stat.size > 32 * 1024 * 1024) throw new Error('Expected a nonempty regular file at most 32 MiB');
  const bytes = readFileSync(file);
  return { path: name, bytes: bytes.length, sha256: hash(bytes) };
}

/** Offline, resource-bounded execution. Scripts/results persist; interpreter
 * variables intentionally do not. No ambient environment or host home is mounted. */
export class WorkerAnalysisRuntime implements AnalysisRuntime {
  constructor(private readonly options: { stateDirectory: string; libraryDirectory: string; binding(session: ProductSession): AnalysisBinding | undefined; worker?(session:ProductSession):SandboxWorker | undefined }) {}
  private scope(session: ProductSession) {
    const binding = this.options.binding(session);
    if (!binding || !session.runId || !binding.scope || !/^sha256:[a-f0-9]{64}$/u.test(binding.image)) throw new Error('No host-authorized, image-pinned analysis workspace for this study');
    if (!(binding.maxExecutions === null || Number.isSafeInteger(binding.maxExecutions) && binding.maxExecutions > 0) || !Number.isSafeInteger(binding.maxSeconds) || binding.maxSeconds <= 0) throw new Error('Valid execution quota and finite per-process timeout required');
    const selected=this.options.worker?.(session);
    if(this.options.worker && !selected)throw new Error('Configured sandbox worker unavailable; no local fallback is authorized');
    const worker=selected ?? new DockerSandboxWorker({image:binding.image});
    const description=worker.describe();
    if (description.image!==binding.image || (this.options.worker || binding.workerFingerprint) && binding.workerFingerprint!==description.fingerprint) throw new Error('Sandbox worker changed or is not authorized: bind the current worker fingerprint before continuing');
    if (description.network!=='none' || description.execution!=='python-script' || !description.persistentFiles || description.persistentVariables!==false || !Number.isFinite(description.maxSeconds) || description.maxSeconds<=0) throw new Error('Incompatible sandbox worker capabilities');
    for (const dir of [binding.workspace, binding.inputs, binding.model, this.options.libraryDirectory]) {
      if (!path.isAbsolute(dir) || !lstatSync(dir).isDirectory() || lstatSync(dir).isSymbolicLink() || realpathSync(dir) !== path.resolve(dir) || dir.includes(',')) throw new Error('Analysis mounts must be real absolute directories without symlinks or commas');
    }
    const key = hash(JSON.stringify({session:session.id,run:session.runId,binding,worker:description.fingerprint}));
    const state = path.join(this.options.stateDirectory, key); mkdirSync(state, { recursive: true, mode: 0o700 });
    const ledgerPath = path.join(state, 'ledger.json');
    const ledger: Ledger = existsSync(ledgerPath) ? JSON.parse(readFileSync(ledgerPath, 'utf8')) : { executions: [] };
    return { binding, worker, description, state, ledger, save: () => writeFileSync(ledgerPath, JSON.stringify(ledger, null, 2), { mode: 0o600 }) };
  }
  inspect(session: ProductSession) {
    const {binding,ledger,description} = this.scope(session);
    const files: FileReceipt[] = [];
    const scan = (dir: string, prefix = '', depth = 0) => {
      if (depth > 3 || files.length >= 60) return;
      for (const item of readdirSync(dir, { withFileTypes: true }).sort((a,b)=>a.name.localeCompare(b.name))) {
        if (files.length >= 60) break;
        // Saved execution scripts already have receipts in the ledger. Do not
        // let this unbounded history crowd out the actual analysis artifacts.
        if (depth === 0 && item.name === 'analysis-scripts') continue;
        const name = prefix + item.name;
        if (item.isDirectory()) scan(path.join(dir,item.name),name+'/',depth+1);
        else if (item.isFile()) { try { files.push(receipt(binding.workspace,name)); } catch { /* Not an eligible deliverable. */ } }
      }
    };
    scan(binding.workspace);
    const delivered = ledger.delivered?.every(old => { try { return receipt(binding.workspace,old.path).sha256 === old.sha256; } catch { return false; } }) ?? false;
    return { scope:binding.scope, worker:description, locations:{input:'/input',model:'/model',work:'/work',library:'/analysis'},
      status:ledger.executions.some(run=>run.worker && run.cleanupConfirmed!==true)?'host_recovery_required':'ready',
      execution:'Fresh Python process; files persist. Every supplied script is automatically saved. Reuse scripts with runpy.run_path; do not rely on previous variables.',
      library:'Import theta_mining for topic probes, exhaustive condition partitions and contrasts. Read /analysis/README.md for contracts and limitations.',
      files, inventoryBounded:true, inventoryExcludes:['analysis-scripts/'],
      savedScripts:ledger.executions.slice(-5).map(run=>run.script),
      executions:ledger.executions.slice(-5), executionsUsed:ledger.executions.length,
      deliveredFiles:ledger.delivered ?? [],
      executionsRemaining:binding.maxExecutions === null ? null : Math.max(0,binding.maxExecutions-ledger.executions.length), delivered,
      instruction:'Receipts verify bytes and execution, not scientific correctness. Agent notes and a saved plan do not prove completion.' };
  }
  async execute(session: ProductSession, request: AnalysisExecution, signal?: AbortSignal) {
    signal?.throwIfAborted();
    const {binding,state,ledger,save,worker,description} = this.scope(session);
    if (binding.maxExecutions !== null && ledger.executions.length >= binding.maxExecutions) throw new Error('Host analysis execution limit exhausted');
    if (ledger.executions.some(run=>run.worker && run.cleanupConfirmed!==true)) throw new Error('Prior sandbox cleanup is unresolved; host recovery is required before executing again');
    if (!request.code.trim() || request.code.length > 60000) throw new Error('Provide a bounded executable analysis script');
    for (const name of request.outputs) workspaceFile(binding.workspace,name);
    const id = randomUUID(), scriptName = `analysis-scripts/${id}.py`;
    const scripts = workspaceFile(binding.workspace,'analysis-scripts'); mkdirSync(scripts,{recursive:true});
    writeFileSync(workspaceFile(binding.workspace,scriptName), request.code, {flag:'wx',mode:0o644});
    // Audit copies are outside the code sandbox, including failed executions.
    writeFileSync(path.join(state,`${id}.py`),request.code,{flag:'wx',mode:0o600});
    const before = new Map(request.outputs.map(name=>{try{return [name,receipt(binding.workspace,name).sha256];}catch{return [name,null];}}));
    const item: RunReceipt = {id,purpose:request.purpose,script:receipt(binding.workspace,scriptName),exitCode:null,outputs:[],changedOutputs:[],missing:[...request.outputs],seconds:0,worker:description};
    ledger.executions.push(item); ledger.delivered = undefined; save();
    let output='',diagnostics='',streamInfo:{}={};
    try {
      const result=await worker.run({id,script:scriptName,workspace:binding.workspace,inputs:binding.inputs,model:binding.model,library:this.options.libraryDirectory,
        seconds:Math.max(1,Math.min(request.seconds,binding.maxSeconds,description.maxSeconds,180))},signal);
      output=result.output.slice(0,12000);diagnostics=result.diagnostics.slice(0,8000);
      streamInfo={outputBytes:result.outputBytes,diagnosticsBytes:result.diagnosticsBytes,outputTruncated:result.outputTruncated,diagnosticsTruncated:result.diagnosticsTruncated};
      Object.assign(item,{exitCode:result.exitCode,seconds:result.seconds,error:result.error,cleanupConfirmed:result.cleanupConfirmed});
      if(!result.cleanupConfirmed)item.error ??= 'Sandbox cleanup is unresolved';
    } catch(error) {item.error=error instanceof Error?error.message:String(error);item.cleanupConfirmed=false;}
    item.missing = [];
    for (const name of request.outputs) { try {const file=receipt(binding.workspace,name);item.outputs.push(file);if(before.get(name)!==file.sha256)item.changedOutputs.push(name);} catch {item.missing.push(name);} }
    save(); writeFileSync(path.join(state,`${id}.json`),JSON.stringify({item,output,diagnostics,...streamInfo}),{mode:0o600});
    return {...item,output,diagnostics,...streamInfo,outputBounded:true,completed:item.exitCode===0 && !item.error && item.missing.length===0,
      instruction:'Script saved automatically. Reuse it via runpy.run_path. A successful process or file hash is not validation of the scientific claim; verify definitions and actual tests.'};
  }
  /** Host-only recovery. No inference tool can clear a failed cleanup receipt. */
  async recover(session:ProductSession, executionId:string) {
    const {worker,ledger,save}=this.scope(session);
    const execution=ledger.executions.find(run=>run.id===executionId);
    if(!execution || !worker.recover)throw new Error('No recoverable owned execution in the current worker scope');
    const confirmed=await worker.recover(executionId);
    execution.cleanupConfirmed=confirmed;save();
    return {executionId,cleanupConfirmed:confirmed,instruction:'Only sandbox cleanup state changed. The original execution error and scientific outcome are unchanged.'};
  }
  deliver(session: ProductSession, files: string[]) {
    const {binding,state,ledger,save} = this.scope(session);
    if(ledger.executions.some(run=>run.worker && run.cleanupConfirmed!==true))throw new Error('Resolve sandbox cleanup before archiving mutable outputs');
    if (!files.length || !ledger.executions.some(r=>r.exitCode===0 && !r.error)) throw new Error('Delivery requires executed analysis and actual files');
    const snapshots = files.map(name=>({receipt:receipt(binding.workspace,name),bytes:readFileSync(workspaceFile(binding.workspace,name))}));
    const id=randomUUID(), destination=path.join(state,`delivery-${id}`); mkdirSync(destination,{mode:0o700});
    snapshots.forEach(value=>{const file=workspaceFile(destination,value.receipt.path);mkdirSync(path.dirname(file),{recursive:true});writeFileSync(file,value.bytes,{flag:'wx',mode:0o600});});
    ledger.delivered=snapshots.map(v=>v.receipt); save();
    writeFileSync(path.join(state,`delivery-${id}.json`),JSON.stringify(ledger.delivered,null,2),{flag:'wx',mode:0o600});
    return {delivered:true,files:ledger.delivered,deliveryId:id,
      artifacts:ledger.delivered.map(file=>({name:file.path,path:path.join(destination,file.path)})),
      instruction:'Original bytes archived. Artifact paths are host download locations, not sandbox paths. This verifies delivery only, not correctness, completeness of reasoning, or an external evaluation score.'};
  }
}

/** Backward-compatible default; new hosts can select a worker per study. */
export class DockerAnalysisRuntime extends WorkerAnalysisRuntime {}
