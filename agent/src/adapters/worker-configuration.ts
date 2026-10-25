import { createRequire } from 'node:module';
import { mkdirSync, existsSync, readFileSync, writeFileSync, renameSync } from 'node:fs';
import path from 'node:path';
import { contentHash } from '../domain/research.js';
const require = createRequire(import.meta.url);
const { createCredentials } = require('../../../runtime/credentials.cjs');
export interface WorkerConfiguration { mode: 'local' | 'remote'; url: string; encryptedToken?: string; maxCost: number; currency: string }
export function safeWorkerUrl(value: string): URL {
  const url = new URL(value);
  if (url.username || url.password || url.search || url.hash || !(url.protocol === 'https:' || url.protocol === 'http:' && ['127.0.0.1','localhost','[::1]'].includes(url.hostname))) throw new Error('计算服务需要 HTTPS；本机测试可使用 HTTP');
  return url;
}
export function workerConfiguration(home: string) {
  const root = process.env.THETA_COMPUTE_SETTINGS_HOME ?? process.env.THETA_DESKTOP_HOME ?? process.env.THETA_AGENT_HOME ?? home;
  mkdirSync(root, { recursive: true });
  const file = path.join(root, 'compute-settings.json');
  const credentials = createCredentials(root);
  const read = (): WorkerConfiguration => existsSync(file) ? JSON.parse(readFileSync(file,'utf8')) : { mode:'local',url:'',maxCost:0,currency:'CNY' };
  const publicValue = () => { const {encryptedToken,...v}=read();return {...v,tokenConfigured:Boolean(encryptedToken),fingerprint:contentHash(read())}; };
  return { read, publicValue, token:(value: WorkerConfiguration)=>value.encryptedToken ? credentials.decrypt(value.encryptedToken) : '',
    fingerprint:(value: WorkerConfiguration)=>contentHash(value),
    save(input: Record<string,unknown>) {
      const previous=read();
      if (!['local','remote'].includes(String(input.mode))) throw new Error('请选择本地或远程计算');
      const url=String(input.url ?? '').trim().replace(/\/+$/,'');
      if(url) safeWorkerUrl(url);
      const maxCost=Number(input.maxCost ?? 0),currency=String(input.currency ?? 'CNY').toUpperCase();
      if(!Number.isFinite(maxCost)||maxCost<0||!['CNY','USD'].includes(currency))throw new Error('费用上限须为非负数，币种为 CNY 或 USD');
      const token=String(input.token ?? '').trim();
      const encryptedToken=input.clearToken ? undefined : token ? credentials.encrypt(token) : previous.encryptedToken;
      if(input.mode==='remote'&&(!url||!encryptedToken))throw new Error('请填写远程计算地址和 API Key');
      const value={mode:input.mode,url,encryptedToken,maxCost,currency};
      writeFileSync(file+'.tmp',JSON.stringify(value),{mode:0o600});renameSync(file+'.tmp',file);
      return publicValue();
    }
  };
}
