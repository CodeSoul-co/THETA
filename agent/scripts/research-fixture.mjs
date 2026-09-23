import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

// Diagnostic corpus only. Deliberately constructed time/group associations are not research findings.
export function writeResearchFixture(file) {
  const themes = [
    ['The bus route needs more frequent service', 'Commuters wait for crowded trains', 'Residents request reliable public transport', 'A new bicycle lane would improve the commute'],
    ['Tenants struggle with rising apartment rent', 'Affordable housing remains difficult to find', 'Residents request repairs in rental buildings', 'Families need stable leases and housing support'],
    ['Households discuss rooftop solar panels', 'Energy bills increased during cold weather', 'Residents request better building insulation', 'Electricity savings depend on efficient heating'],
  ];
  const contexts = ['near the central station', 'in the western neighborhood', 'around the university district', 'in the northern suburb', 'beside the public library'];
  const reasons = ['Access matters for daily routines.', 'The current arrangement requires careful planning.', 'Participants compare several local options.', 'The discussion includes costs and practical tradeoffs.', 'Residents describe their recent observations.', 'The proposal would require further review.'];
  const rows = [];
  for (let year = 2023; year <= 2025; year++) {
    const counts = [[30, 20, 10], [20, 20, 20], [10, 20, 30]][year - 2023];
    for (let topic = 0; topic < 3; topic++) for (let i = 0; i < counts[topic]; i++) {
      const text = `${themes[topic][i % 4]} ${contexts[Math.floor(i / 4) % 5]}. ${reasons[Math.floor(i / 20) + year - 2023]} ${i % 7 === 0 ? themes[(topic + 1) % 3][(i + 1) % 4] + '.' : ''}`.trim();
      rows.push({doc_id:`synthetic-${year}-${topic}-${i}`,text,year,region:i % 3 === topic % 3 ? 'north' : 'south',channel:i % 2 ? 'forum' : 'survey',synthetic:true});
    }
  }
  // Interleave years to exercise source-row/time alignment, deterministically.
  const shuffled = Array.from({length:60}, (_, i) => [rows[i], rows[i+60], rows[i+120]]).flat();
  const columns = Object.keys(shuffled[0]);
  const quote = value => `"${String(value).replaceAll('"','""')}"`;
  mkdirSync(path.dirname(file), {recursive:true});
  writeFileSync(file, [columns.join(','), ...shuffled.map(row => columns.map(key => quote(row[key])).join(','))].join('\n') + '\n');
  return {file,rows:shuffled.length,uniqueTexts:new Set(shuffled.map(row=>row.text)).size,yearCounts:{2023:60,2024:60,2025:60}};
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) console.log(writeResearchFixture(process.argv[2] || '/tmp/theta-research-synthetic.csv'));
