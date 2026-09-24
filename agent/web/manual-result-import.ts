import { createHash, randomUUID } from 'node:crypto';
import { constants, createReadStream, existsSync, lstatSync, realpathSync } from 'node:fs';
import { cp, mkdir, readdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import type { ComputeJob, Dataset, ResearchRun } from '../src/domain/research.js';
import { contentHash } from '../src/domain/research.js';
import type { ComputeRequest } from '../src/adapters/compute-gateway.js';
import type { ProductSessionStore, ProductSession } from '../src/memory/session-store.js';
import type { ResearchStore } from '../src/memory/research-store.js';

export class ManualImportError extends Error {
  constructor(readonly status: number, message: string) { super(message); }
}
interface ManualProject { id: number | string; name: string; dataset_name: string; archived?: boolean }
interface ImportInput { projectId?: number; datasetName?: string; selectedJobId?: string }
interface ManualJob { dataset_name: string; workers?: Array<{ id: string; model: string }> }
interface ManualFile { dataset_name: string; dataset: Dataset }
export interface ImportedProject {
  project: { id: string; name: string; createdAt: string; updatedAt: string; pinned: boolean; runIds: string[] };
  runId: string; copiedJobs: number; reused: boolean;
}

const inside = (root: string, file: string): string => {
  const resolved = realpathSync.native(file);
  const relative = path.relative(realpathSync.native(root), resolved);
  if (!relative || relative.startsWith('..') || path.isAbsolute(relative) || lstatSync(file).isSymbolicLink()) {
    throw new ManualImportError(409, '项目文件路径无效，未复制结果。');
  }
  return resolved;
};
const copy = async (source: string, target: string) => {
  await mkdir(path.dirname(target), { recursive: true });
  // Reflinks are independent copies, unlike hard links: later edits cannot change the source.
  await cp(source, target, { recursive: true, force: false, errorOnExist: true, mode: constants.COPYFILE_FICLONE,
    filter: file => { if (lstatSync(file).isSymbolicLink()) throw new ManualImportError(409, '项目中包含符号链接，不能安全复制。'); return true; } });
};
const inventory = async (root: string, directory = root): Promise<Array<{ name: string; path: string; kind: string }>> => {
  const entries = await readdir(directory, { withFileTypes: true });
  return (await Promise.all(entries.map(async entry => {
    const file = path.join(directory, entry.name);
    if (entry.isDirectory()) return inventory(root, file);
        const kind = /\.(png|jpe?g|svg|pdf|webp)$/iu.test(file) ? 'figure' : /\.(csv|tsv)$/iu.test(file) ? 'table' : /\.npy$/iu.test(file) ? 'matrix' : /\.html$/iu.test(file) ? 'report' : 'artifact';
    return [{ name: path.relative(root, file).split(path.sep).join('/'), path: file, kind }];
  }))).flat();
};
async function fileHash(file: string): Promise<string> {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest('hex');
}
async function resultHash(files: Awaited<ReturnType<typeof inventory>>): Promise<string> {
  const digest = createHash('sha256');
  // Match pathlib's component-wise ordering used by the worker's tree_hash.
  for (const file of [...files].sort((a, b) => {
    const left = a.name.split('/'); const right = b.name.split('/');
    for (let i = 0; i < Math.min(left.length, right.length); i++) {
      const order = Buffer.compare(Buffer.from(left[i]), Buffer.from(right[i]));
      if (order) return order;
    }
    return left.length - right.length;
  })) {
    digest.update(file.name).update('\0').update(await fileHash(file.path)).update('\n');
  }
  return digest.digest('hex');
}

/** Local-only bridge at the completed-result boundary; never runs training or inference. */
export function manualResultImporter(home: string, manualHome: string, records: ResearchStore, sessions: ProductSessionStore) {
  const pending = new Map<string, Promise<ImportedProject>>();
  let queue: Promise<unknown> = Promise.resolve();
  return (input: ImportInput, ownerId: string): Promise<ImportedProject> => {
    const key = JSON.stringify([ownerId, input.projectId, input.datasetName]);
    const existing = pending.get(key);
    if (existing) return existing;
    // Serialize commits, including legacy dataset and project-ID aliases of the same source.
    const operation = queue.then(() => importProject(input, ownerId)).finally(() => pending.delete(key));
    queue = operation.catch(() => undefined);
    pending.set(key, operation);
    return operation;
  };

  async function importProject(input: ImportInput, ownerId: string): Promise<ImportedProject> {
    if (input.projectId !== undefined ? !Number.isSafeInteger(input.projectId) || input.projectId <= 0
      : typeof input.datasetName !== 'string' || !input.datasetName.trim() || input.datasetName.length > 160) throw new ManualImportError(400, '请选择有效的手动项目。');
    if (!existsSync(path.join(manualHome, 'manual.sqlite'))) throw new ManualImportError(404, '手动项目存储不存在。');
    const source = new DatabaseSync(path.join(manualHome, 'manual.sqlite'), { readOnly: true });
    let project: ManualProject;
    let files: ManualFile[];
    let jobs: Array<{ id: string; request: ComputeRequest; state: ComputeJob }> = [];
    try {
      if (input.projectId !== undefined) {
        const row = source.prepare("SELECT value FROM records WHERE kind='project' AND id=?").get(input.projectId);
        if (!row) throw new ManualImportError(404, '手动项目不存在。');
        project = { ...JSON.parse(String(row.value)), id: input.projectId };
      } else {
        // Older local datasets appear as projects in the UI without a project row.
        const projects = source.prepare("SELECT id,value FROM records WHERE kind='project'").all().map(row => ({ ...JSON.parse(String(row.value)), id: Number(row.id) } as ManualProject));
        project = projects.find(project => project.dataset_name === input.datasetName) ?? { id: `dataset:${input.datasetName}`, name: input.datasetName!, dataset_name: input.datasetName! };
      }
      if (project.archived) throw new ManualImportError(404, '手动项目已归档。');
      const list = <T>(kind: string): T[] => source.prepare('SELECT value FROM records WHERE kind=? ORDER BY id DESC').all(kind).map(row => JSON.parse(String(row.value)));
      files = list<ManualFile>('file').filter(file => file.dataset_name === project.dataset_name);
      const ids = [...new Set(list<ManualJob>('job').filter(job => job.dataset_name === project.dataset_name).flatMap(job => job.workers?.map(worker => worker.id) ?? []))];
      if (ids.length && existsSync(path.join(manualHome, 'compute.sqlite'))) {
        const compute = new DatabaseSync(path.join(manualHome, 'compute.sqlite'), { readOnly: true });
        try {
          jobs = ids.flatMap(id => {
            const row = compute.prepare("SELECT request,value FROM jobs WHERE id=? AND state='completed'").get(id);
            return row ? [{ id, request: JSON.parse(String(row.request)), state: JSON.parse(String(row.value)) }] : [];
          });
        } finally { compute.close(); }
      }
    } finally { source.close(); }
    if (!jobs.length) throw new ManualImportError(409, '当前项目还没有已完成的训练结果，不能复制到对话模式。');
    if (input.selectedJobId && !jobs.some(job => job.id === input.selectedJobId)) throw new ManualImportError(409, '所选结果尚未完成或不属于当前项目。');
    const fingerprint = contentHash({ ownerId, projectId: project.id, jobs: jobs.map(job => [job.id, job.state.resultHash]).sort(), datasets: files.map(file => file.dataset.sha256).sort() });
    try {
      const previous = records.get<ImportedProject>('manual-result-import', fingerprint);
      const savedProject = records.get<{ archived?: boolean }>('web-project', previous.project.id);
      const savedSession = records.get<{ archived?: boolean }>('web-session', previous.runId);
      if (!savedProject.archived && !savedSession.archived) return { ...previous, reused: true };
    } catch { /* No successful import yet. */ }

    const projectId = randomUUID();
    const importRoot = path.join(home, 'imports', projectId);
    const copiedRoots: string[] = [importRoot];
    const datasets = new Map<string, Dataset>();
    const runs: ResearchRun[] = [];
    const reports: NonNullable<ProductSession['reports']> = [];
    const computeRows: Array<{ id: string; request: ComputeRequest; state: ComputeJob }> = [];
    const now = new Date().toISOString();
    try {
      for (const original of [...files.map(file => file.dataset), ...jobs.map(job => job.request.dataset)]) {
        if (datasets.has(original.datasetRef)) continue;
        const managedPath = path.join(importRoot, 'datasets', contentHash(original), path.basename(original.fileName));
        await copy(inside(manualHome, original.managedPath), managedPath);
        if (await fileHash(managedPath) !== original.sha256) throw new ManualImportError(409, '数据集文件与上传时的校验记录不一致，未创建对话副本。');
        datasets.set(original.datasetRef, { ...original, datasetRef: `dataset-${randomUUID()}`, managedPath });
      }
      for (const job of jobs) {
        if (!/^job-[a-f0-9]{64}$/u.test(job.id) || !job.state.resultDir) throw new ManualImportError(409, '训练结果记录不完整。');
        const sourceRoot = inside(manualHome, path.join(manualHome, 'compute', job.id));
        const sourceResult = inside(sourceRoot, job.state.resultDir);
        const jobId = `job-${contentHash({ projectId, sourceJobId: job.id })}`;
        const targetRoot = path.join(home, 'compute', jobId);
        copiedRoots.push(targetRoot);
        await copy(sourceRoot, targetRoot);
        const resultDir = path.join(targetRoot, path.relative(sourceRoot, sourceResult));
        const state: ComputeJob = { ...job.state, id: jobId, status: 'completed', resultDir, pid: undefined };
        const run: ResearchRun = { id: `run-${randomUUID()}`, goal: `${project.name}：继续解读与修改 ${job.request.plan.modelId} 结果`,
          datasetRef: datasets.get(job.request.dataset.datasetRef)!.datasetRef, computeBackend: 'local', plan: job.request.plan,
          planHash: contentHash(job.request.plan), jobs: [jobId], activeJob: jobId, lastObservedJob: state,
          notes: [`由手动项目 ${project.id} 的已完成任务 ${job.id} 复制；原项目不变，未重新训练。`] };
        runs.push(run);
        computeRows.push({ id: jobId, state, request: { jobId, runId: run.id, dataset: datasets.get(job.request.dataset.datasetRef)!, plan: job.request.plan } });
        const delivered = await inventory(resultDir);
        if (!delivered.length || await resultHash(delivered) !== job.state.resultHash) {
          throw new ManualImportError(409, '结果文件与训练完成时的校验记录不一致，未创建对话副本。');
        }
        // Editing uses a report-owned copy; compute.results keeps its immutable hash.
        // Keep plotting tables under the report root, as required by figure.adjust.
        const reportRoot = path.join(importRoot, 'reports', jobId);
        await copy(resultDir, reportRoot);
        const reportFiles = await inventory(reportRoot);
        const reportPath = path.join(reportRoot, 'continuation.html');
        const nativeReport = reportFiles.find(file => file.name === 'zh/index.html') ?? reportFiles.find(file => file.name === 'index.html');
        await writeFile(reportPath, '<!doctype html><meta charset="utf-8"><title>手动项目结果副本</title><h1>已复制的训练结果</h1><p>原手动项目未改动；图表、绘图数据与模型文件也可在研究结果页下载。</p>' +
          (nativeReport ? `<p><a href="${encodeURI(nativeReport.name)}">查看原生可视化报告</a></p>` : '<p>原任务未生成可视化报告，当前仅保留实际存在的训练产物。</p>'));
        reports.push({ jobId, reportPath, files: reportFiles });
      }
      // Install only terminal records; no approval, queue, PID or worker is carried over.
      const compute = new DatabaseSync(path.join(home, 'compute.sqlite'));
      try {
        compute.exec('PRAGMA busy_timeout=5000; CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, request TEXT NOT NULL, state TEXT NOT NULL, value TEXT NOT NULL, cancel INTEGER NOT NULL DEFAULT 0, updated REAL NOT NULL); BEGIN IMMEDIATE');
        for (const row of computeRows) compute.prepare('INSERT INTO jobs(id,request,state,value,updated) VALUES (?,?,?,?,?)').run(row.id, JSON.stringify(row.request), 'completed', JSON.stringify(row.state), Date.now() / 1000);
        compute.exec('COMMIT');
      } catch (error) { compute.exec('ROLLBACK'); throw error; }
      finally { compute.close(); }
      const session = sessions.create();
      session.title = `${project.name} · 结果讨论`;
      session.datasetRefs = [...datasets.values()].map(dataset => dataset.datasetRef);
      session.runIds = runs.map(run => run.id);
      const selectedIndex = Math.max(0, jobs.findIndex(job => job.id === input.selectedJobId));
      session.runId = runs[selectedIndex].id;
      session.reports = reports;
      session.resultSelection = { mode: 'manual', resolvedJobIds: [computeRows[selectedIndex].id], resolvedAt: now };
      session.messages = [{ role: 'assistant', content: `已从手动项目「${project.name}」复制 ${jobs.length} 个已完成的模型结果（${jobs.map(job => job.request.plan.modelId).join('、')}），并保留数据集、模型参数、图表、绘图数据和训练模型。\n\n这是独立的对话项目；原手动项目未改动，也没有重新训练。你可以直接询问结果含义、比较模型或要求调整图表。需要新训练时仍会先请你确认。`, metadata: { webCreatedAt: now } }];
      sessions.save(session);
      const importedProject = { id: projectId, name: `${project.name} · 对话副本`, createdAt: now, updatedAt: now, pinned: false, runIds: [session.id] };
      const result = { project: importedProject, runId: session.id, copiedJobs: jobs.length, reused: false };
      records.putMany([
        ...[...datasets.values()].flatMap(dataset => [
          { kind: 'dataset', id: dataset.datasetRef, value: dataset },
          { kind: 'web-dataset-owner', id: dataset.datasetRef, value: { datasetRef: dataset.datasetRef, ownerId, projectIds: [projectId] } },
        ]),
        ...runs.map(run => ({ kind: 'run', id: run.id, value: run })),
        { kind: 'web-project', id: projectId, value: { ...importedProject, ownerId } },
        { kind: 'web-session', id: session.id, value: { id: session.id, projectId, ownerId, createdAt: now, updatedAt: now, pinned: false } },
        { kind: 'manual-result-import', id: fingerprint, value: result },
      ]);
      return result;
    } catch (error) {
      if (computeRows.length && existsSync(path.join(home, 'compute.sqlite'))) {
        const compute = new DatabaseSync(path.join(home, 'compute.sqlite'));
        try { for (const row of computeRows) compute.prepare('DELETE FROM jobs WHERE id=?').run(row.id); }
        finally { compute.close(); }
      }
      // Only newly allocated copy directories are removed; original data is never touched.
      for (const directory of copiedRoots) await rm(directory, { recursive: true, force: true });
      throw error;
    }
  }
}
