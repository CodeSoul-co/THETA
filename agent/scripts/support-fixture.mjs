import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';

// Synthetic fixture shared by real-worker and simulated-gateway acceptance.
export function writeSupportFixture(file) {
  mkdirSync(path.dirname(file), { recursive: true });
  const rows = ['text,channel,timestamp'];
  const samples = [
    'parcel delivery arrived late tracking shipment courier delayed package shipping service',
    'refund payment billing charged twice invoice money return customer support response',
    'application login password account crashes software update screen authentication error',
  ];
  for (let i = 0; i < 36; i++) rows.push(`${samples[i % 3]} ${['help request', 'problem unresolved', 'issue resolved', 'followup contact'][i % 4]},${i % 2 ? 'phone' : 'web'},2026-08-${String(i % 28 + 1).padStart(2, '0')}`);
  writeFileSync(file, rows.join('\n'));
  return file;
}
