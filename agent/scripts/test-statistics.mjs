import {spawnSync} from 'node:child_process';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const python=process.env.THETA_WORKER_STATISTICS_PYTHON||path.join(root,'.local/runtimes/statistics/bin/python');
const result=spawnSync(python,['-m','unittest','workers.statistics.test_statistics','-v'],{cwd:root,stdio:'inherit',env:{...process.env,OMP_NUM_THREADS:'1',OPENBLAS_NUM_THREADS:'1',MKL_NUM_THREADS:'1',PYTHONNOUSERSITE:'1'}});
if(result.error)console.error(`Statistics environment unavailable: ${result.error.message}. See docs/free-analysis.md.`);
process.exit(result.status??1);
