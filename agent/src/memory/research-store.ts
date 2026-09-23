import { mkdirSync } from 'node:fs';
import { DatabaseSync } from 'node:sqlite';
import path from 'node:path';

/** Agent-owned research records only. Compute task state remains in its gateway. */
export class ResearchStore {
  constructor(readonly directory: string) { mkdirSync(directory, { recursive: true }); }
  private use<T>(fn: (db: DatabaseSync) => T): T {
    const db = new DatabaseSync(path.join(this.directory, 'research.sqlite'));
    try {
      db.exec('PRAGMA busy_timeout=5000; CREATE TABLE IF NOT EXISTS records (kind TEXT NOT NULL, id TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY(kind,id))');
      return fn(db);
    } finally { db.close(); }
  }
  get<T>(kind: string, id: string): T {
    return this.use((db) => {
      const row = db.prepare('SELECT value FROM records WHERE kind=? AND id=?').get(kind, id);
      if (!row) throw new Error(`记录不存在：${kind}/${id}`);
      return JSON.parse(String(row.value)) as T;
    });
  }
  put(kind: string, id: string, value: unknown): void {
    this.use((db) => { db.prepare('INSERT INTO records VALUES (?,?,?) ON CONFLICT(kind,id) DO UPDATE SET value=excluded.value').run(kind, id, JSON.stringify(value)); });
  }
  /** Publish a related set of records atomically (e.g. a copied project and its runs). */
  putMany(entries: Array<{ kind: string; id: string; value: unknown }>): void {
    this.use(db => {
      db.exec('BEGIN IMMEDIATE');
      try {
        const statement = db.prepare('INSERT INTO records VALUES (?,?,?) ON CONFLICT(kind,id) DO UPDATE SET value=excluded.value');
        for (const entry of entries) statement.run(entry.kind, entry.id, JSON.stringify(entry.value));
        db.exec('COMMIT');
      } catch (error) { db.exec('ROLLBACK'); throw error; }
    });
  }
  list<T>(kind: string, offset = 0): T[] {
    return this.use(db => db.prepare('SELECT value FROM records WHERE kind=? ORDER BY rowid DESC LIMIT 100 OFFSET ?').all(kind, offset).map(row => JSON.parse(String(row.value)) as T));
  }
  putIfAbsent(kind: string, id: string, value: unknown): boolean {
    return this.use((db) => db.prepare('INSERT OR IGNORE INTO records VALUES (?,?,?)').run(kind, id, JSON.stringify(value)).changes === 1);
  }
  compareAndSet(kind: string, id: string, expected: unknown, value: unknown): boolean {
    return this.use((db) => db.prepare('UPDATE records SET value=? WHERE kind=? AND id=? AND value=?').run(JSON.stringify(value), kind, id, JSON.stringify(expected)).changes === 1);
  }
}
