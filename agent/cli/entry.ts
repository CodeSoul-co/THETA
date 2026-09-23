import path from 'node:path';
import { loadThetaProjectEnvironment, repositoryRoot } from '../src/environment.js';
import { configuredProviderSummaries } from '../src/providers/configured-provider.js';
import { PythonCapabilityWorker } from '../src/adapters/python-worker.js';
import { AgentShell } from './shell/agent-shell.js';
import { runToolsCli } from './commands/tools-cli.js';

export async function main(args: string[]): Promise<number> {
  try {
    loadThetaProjectEnvironment();
    if (args[0] === '--help' || args[0] === '-h') {
      console.log('THETA · 主题与自由分析 Agent\n\n运行 theta 开始持续对话，无需输入训练参数。\n\ntheta [interactive] [--session <id>]\ntheta tools list\ntheta tools call <name> --session <id>  （stdin JSON）\ntheta doctor [--json]\n\n会话内：/mode free|topic /attach /new /sessions /resume /model /models /help /exit'); return 0;
    }
    if (args[0] === 'tools') return runToolsCli(args.slice(1), { write: (text) => console.log(text), writeError: (text) => console.error(text) });
    if (args[0] === 'doctor') {
      let compute: unknown;
      try { compute = await new PythonCapabilityWorker().call('runtime.check', { modelId: 'lda' }); }
      catch (error) { compute = { ready: false, error: error instanceof Error ? error.message : String(error) }; }
      const worker = new PythonCapabilityWorker();
      const environments = await Promise.all(['classic', 'neural', 'reports', 'statistics'].map(async profile => {
        try { return await worker.call('runtime.environment', { profile }); }
        catch (error) { return { profile, available: false, error: error instanceof Error ? error.message : String(error) }; }
      }));
      console.log(JSON.stringify({ agent: 'theta-agent', node: process.version, engine: repositoryRoot(), home: path.resolve(process.env.THETA_AGENT_HOME ?? '.theta_agent'), computeMode: process.env.THETA_BUSINESS_API_URL ? 'business' : process.env.THETA_COMPUTE_URL ? 'go' : 'local', providers: configuredProviderSummaries(), compute, environments }, null, 2)); return 0;
    }
    const remaining = args[0] === 'interactive' ? args.slice(1) : args;
    if (remaining.length && !(remaining.length === 2 && remaining[0] === '--session')) throw new Error('未识别参数；运行 theta --help 查看入口，或直接运行 theta 开始对话。');
    return new AgentShell({ sessionId: remaining[1] }).run();
  } catch (error) { console.error(error instanceof Error ? error.message : String(error)); return 1; }
}
