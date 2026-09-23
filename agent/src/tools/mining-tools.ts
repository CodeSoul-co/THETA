import { z } from 'zod';
import type { AnalysisRuntime } from '../adapters/analysis-runtime.js';
import { notebookKey, type ProductSession } from '../memory/session-store.js';
import type { InferenceProvider } from '../providers/types.js';
import { reviewMethod, methodPlanHash } from './method-review.js';
import { contentHash } from '../domain/research.js';

const file = z.string().min(1).max(240).regex(/^(?!\/)(?!.*(?:^|\/)\.\.?(?:\/|$))[^\\]+$/u);
export const miningPlanSchema = z.object({
  question:z.string().min(1).max(1500),
  candidates:z.array(z.object({
    hypothesis:z.string().min(1).max(800), topicEvidence:z.array(z.string().max(500)).max(6),
    inclusion:z.string().min(1).max(800), exclusion:z.string().min(1).max(800), unknown:z.string().min(1).max(800),
    test:z.string().min(1).max(1000), falsification:z.string().min(1).max(1000),
  }).strict()).min(1).max(3),
  requiredFiles:z.array(file).min(1).max(12),
  stoppingRule:z.string().min(1).max(1000),
}).strict();
export type MiningPlan = z.infer<typeof miningPlanSchema>;
export const miningTools = [
  {name:'analysis_workspace',worker:'analysis',effect:'read',description:'Inspect the host-authorized post-model workspace, persistent files, exact execution receipts and delivery state. Returns actual sandbox locations and the generic theta_mining library. Does not train or grant access.',schema:z.object({}).strict()},
  {name:'analysis_plan',worker:'analysis',effect:'write',description:'Record up to three falsifiable candidate findings AFTER inspecting topic evidence. Define inclusion, exclusion, unknowns, actual test, counterevidence and required delivery files. This is a plan, not a verified conclusion. Preserve user-required deliverables; do not weaken the contract to claim completion.',schema:miningPlanSchema},
  {name:'analysis_execute',worker:'analysis',effect:'effect',description:'Execute your Python analysis in the explicitly host-authorized offline workspace. Use stage=explore to inspect model artifacts and original documents BEFORE choosing hypotheses; no analysis_plan is required for exploration. Use stage=test for planned hypothesis tests. Code is AUTOMATICALLY saved as a reusable script and audited outside the sandbox. Files persist; variables do not. Reuse scripts with runpy.run_path. Declare expected output files. Import theta_mining for general topic probes, condition partitions and sensitivity statistics; hypotheses and semantic labels remain yours.',schema:z.object({stage:z.enum(['explore','test']).default('test'),purpose:z.string().min(1).max(1000),code:z.string().min(1).max(60000),outputs:z.array(file).max(12),seconds:z.number().int().min(1).max(180).default(120)}).strict()},
  {name:'analysis_deliver',worker:'analysis',effect:'write',description:'Verify and archive the original bytes of every required file in the current analysis plan. Requires real execution and nonempty files. Delivery is not a scientific-quality score or external submission. Use any separately provided public validator and submission tool as required by the user.',schema:z.object({}).strict()},
] as const;

export function analysisReady(session: ProductSession): boolean {
  if (session.analysisMode === 'free') return !session.pendingSynthesis && !session.pendingConfirmation;
  return !!session.runId && !session.pendingSynthesis && !session.pendingConfirmation
    && (session.reports ?? []).some(report => report.reportStatus==='complete'
      && (session.interpretations ?? []).some(value=>value.jobIds.includes(report.jobId)));
}
export async function executeMining(runtime: AnalysisRuntime | undefined, name:string, input:unknown,
  context:{session:ProductSession;save():void;signal?:AbortSignal}, reviewer?:InferenceProvider) {
  const definition = miningTools.find(tool=>tool.name===name);
  if (!definition) throw new Error('Unknown analysis tool');
  if (!runtime) throw new Error('Host has not configured or authorized an analysis runtime');
  if (!analysisReady(context.session)) throw new Error('Complete the authorized model report and interpretation, and resolve pending approvals first');
  // The runtime also verifies a host grant bound to the currently selected study.
  runtime.inspect(context.session);
  const args=definition.schema.parse(input);
  if (name==='analysis_workspace') return runtime.inspect(context.session);
  const key=notebookKey(context.session);
  if (name==='analysis_plan') {
    context.session.miningPlans ??= {};
    context.session.miningPlans[key]=args as MiningPlan; context.save();
    if(reviewer){
      const objective=context.session.lastUserIntent ?? context.session.miningPlans[key].question;
      context.session.methodReviews ??= {};
      const previous=context.session.methodReviews[key];
      if(previous?.planHash!==methodPlanHash(args as MiningPlan) || previous.objectiveHash!==contentHash(objective)){
        delete context.session.methodReviews[key];context.save();
        context.session.methodReviews[key]=await reviewMethod(reviewer,context.session.runId!,objective,args as MiningPlan,context.signal);context.save();
      }
    }
    return {saved:true,methodReview:context.session.methodReviews?.[key] ?? null,instruction:'Plan saved; no hypothesis has been tested. If the method review requests revision, address its mismatch with the original objective before testing. Exploration remains available. An aligned method is not verified labels or a scientific result.'};
  }
  if(name==='analysis_execute' && (args as {stage:string}).stage==='explore')return runtime.execute(context.session,args as z.infer<typeof miningTools[2]['schema']>,context.signal);
  const plan=context.session.miningPlans?.[key];
  if (!plan) throw new Error('Record an analysis_plan with observable definitions and required files first');
  if(reviewer){
    const review=context.session.methodReviews?.[key],objective=context.session.lastUserIntent ?? plan.question;
    if(review?.planHash!==methodPlanHash(plan) || review.objectiveHash!==contentHash(objective) || review.verdict!=='aligned')throw new Error('The current plan has no aligned method review. Revise analysis_plan against the original research objective; use stage=explore for additional evidence, not to claim a tested result.');
  }
  if (name==='analysis_execute') return runtime.execute(context.session,args as z.infer<typeof miningTools[2]['schema']>,context.signal);
  const result=runtime.deliver(context.session,plan.requiredFiles) as {artifacts?:Array<{name:string;path:string}>};
  if (result.artifacts) {context.session.pendingArtifacts=[...(context.session.pendingArtifacts??[]),...result.artifacts];context.save();}
  return result;
}
