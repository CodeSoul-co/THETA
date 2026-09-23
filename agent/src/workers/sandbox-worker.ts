import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { OutputCapture } from './output-capture.js';

export interface SandboxDescription {
  id: string; revision: string; fingerprint: string; backend: string;
  image: string; cpus: number; memoryMiB: number; maxSeconds: number;
  network: 'none'; execution: 'python-script'; persistentFiles: true; persistentVariables: false;
}
export interface SandboxExecution {
  id: string; script: string; workspace: string; inputs: string; model: string; library: string; seconds: number;
}
export interface SandboxResult {
  exitCode: number | null; output: string; diagnostics: string; seconds: number;
  error?: string; cleanupConfirmed: boolean;
  outputBytes?: number; diagnosticsBytes?: number; outputTruncated?: boolean; diagnosticsTruncated?: boolean;
}
/** Trusted host port. Implementations own execution and cleanup. A worker must
 * change its fingerprint whenever backend, image, isolation or limits change. */
export interface SandboxWorker {
  describe(): SandboxDescription;
  run(request: SandboxExecution, signal?: AbortSignal): Promise<SandboxResult>;
  recover?(executionId:string):Promise<boolean>;
}

export class DockerSandboxWorker implements SandboxWorker {
  private readonly description: SandboxDescription;
  constructor(options: {image:string; id?:string; cpus?:number; memoryMiB?:number; maxSeconds?:number}) {
    const profile = {id:options.id ?? 'local-docker',revision:'offline-python-3',backend:'docker',image:options.image,
      cpus:options.cpus ?? 4,memoryMiB:options.memoryMiB ?? 6144,maxSeconds:options.maxSeconds ?? 180,
      network:'none' as const,execution:'python-script' as const,persistentFiles:true as const,persistentVariables:false as const};
    if (!/^sha256:[a-f0-9]{64}$/u.test(profile.image) || !profile.id.trim()
      || !Number.isInteger(profile.cpus) || profile.cpus<1 || profile.cpus>32
      || !Number.isInteger(profile.memoryMiB) || profile.memoryMiB<128 || profile.memoryMiB>65536
      || !Number.isInteger(profile.maxSeconds) || profile.maxSeconds<1 || profile.maxSeconds>180) throw new Error('Invalid bounded sandbox worker profile');
    this.description = {...profile,fingerprint:createHash('sha256').update(JSON.stringify(profile)).digest('hex')};
  }
  describe(): SandboxDescription { return {...this.description}; }
  async run(request: SandboxExecution, signal?: AbortSignal): Promise<SandboxResult> {
    signal?.throwIfAborted();
    if (!/^[a-f0-9-]{36}$/u.test(request.id) || !/^analysis-scripts\/[a-f0-9-]{36}\.py$/u.test(request.script)
      || !Number.isFinite(request.seconds) || request.seconds<=0) throw new Error('Invalid sandbox execution request');
    const profile=this.description, container=`theta-analysis-${request.id}`;
    const args=['run','--rm','--pull=never','--name',container,'--network','none','--cpus',String(profile.cpus),
      '--memory',`${profile.memoryMiB}m`,'--memory-swap',`${profile.memoryMiB}m`,'--pids-limit','256',
      '--cap-drop','ALL','--security-opt','no-new-privileges','--read-only','--user','1000:1000','--tmpfs','/tmp:rw,nosuid,size=512m'];
    for(const [src,dst,readOnly] of [[request.inputs,'/input',true],[request.model,'/model',true],[request.workspace,'/work',false],[request.library,'/analysis',true]] as const) args.push('--mount',`type=bind,src=${src},dst=${dst}${readOnly?',readonly':''}`);
    // env -i also discards image-baked environment variables, not only host env.
    args.push('--workdir','/work','--entrypoint','/usr/bin/env',profile.image,'-i','PATH=/usr/local/bin:/usr/bin:/bin','HOME=/tmp',
      'PYTHONPATH=/analysis:/input','PYTHONDONTWRITEBYTECODE=1',`OMP_NUM_THREADS=${profile.cpus}`,`OPENBLAS_NUM_THREADS=${profile.cpus}`,
      `MKL_NUM_THREADS=${profile.cpus}`,`NUMEXPR_NUM_THREADS=${profile.cpus}`,
      'python','-u',`/work/${request.script}`);
    const start=Date.now();
    const result:SandboxResult={exitCode:null,output:'',diagnostics:'',seconds:0,cleanupConfirmed:false};
    const outputCapture=new OutputCapture(12000),diagnosticCapture=new OutputCapture(8000);
    await new Promise<void>(resolve=>{
      const child=spawn('docker',args,{stdio:['ignore','pipe','pipe']});
      const stop=(error:string)=>{result.error ??= error;child.kill('SIGKILL');};
      const abort=()=>stop('Analysis cancelled');
      const timer=setTimeout(()=>stop('Analysis timed out'),Math.min(request.seconds,profile.maxSeconds)*1000);
      let bytes=0;
      const collect=(chunk:Buffer,diagnostic:boolean)=>{bytes+=chunk.length;(diagnostic?diagnosticCapture:outputCapture).push(chunk);if(bytes>2*1024*1024)stop('Analysis output limit exceeded');};
      child.stdout.on('data',c=>collect(c,false));child.stderr.on('data',c=>collect(c,true));
      signal?.addEventListener('abort',abort,{once:true});if(signal?.aborted)abort();
      child.on('error',error=>{result.error ??= error.message;});
      child.on('close',code=>{clearTimeout(timer);signal?.removeEventListener('abort',abort);result.exitCode=code;resolve();});
    });
    const out=outputCapture.finish(),err=diagnosticCapture.finish();
    Object.assign(result,{output:out.text,diagnostics:err.text,outputBytes:out.bytes,diagnosticsBytes:err.bytes,outputTruncated:out.truncated,diagnosticsTruncated:err.truncated});
    // Docker client cancellation does not guarantee container cancellation.
    // rm is scoped to this invocation; inspect must confirm absence afterward.
    result.cleanupConfirmed=await this.recover(request.id);
    if(!result.cleanupConfirmed)result.error ??= 'Sandbox cleanup could not be confirmed; do not start another execution';
    result.seconds=(Date.now()-start)/1000;
    return result;
  }
  async recover(executionId:string):Promise<boolean> {
    if(!/^[a-f0-9-]{36}$/u.test(executionId))throw new Error('Invalid owned execution identifier');
    const container=`theta-analysis-${executionId}`;
    await command(['rm','-f',container]);
    const inspection=await command(['container','ls','--all','--filter',`name=^/${container}$`,'--format','{{.Names}}']);
    return inspection.code===0 && inspection.output.trim()==='';
  }
}

async function command(args:string[]):Promise<{code:number|null;output:string}> {
  return new Promise(resolve=>{
    const child=spawn('docker',args,{stdio:['ignore','pipe','ignore'],timeout:10000,killSignal:'SIGKILL'});let output='';
    child.stdout.on('data',chunk=>{output=(output+chunk.toString()).slice(0,2000);});
    child.on('error',()=>resolve({code:null,output}));child.on('close',code=>resolve({code,output}));
  });
}
