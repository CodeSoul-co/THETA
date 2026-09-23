import { useEffect, useState } from 'react'
import { ActionApprovalCard } from './ActionApprovalCard.tsx'
import { EmbeddingChoiceCard } from './EmbeddingChoiceCard.tsx'
import { TrainingConfigurationCard } from './TrainingConfigurationCard.tsx'
import {
  getStatus,
  postAction,
  type WebAgentInteraction,
  type WebReasoning,
  type WebRunStatus,
} from '../api/client.ts'
import { Button } from '../ui/index.ts'
import { usePreferences } from '../preferences.tsx'
import css from '../styles/app.module.css'

interface RoleColumn { column: string }
interface DatasetUnderstanding {
  domain?: { label?: string }
  analysisUnit?: string
  textColumns?: RoleColumn[]
  timeColumns?: RoleColumn[]
  idColumns?: RoleColumn[]
  metadataColumns?: RoleColumn[]
  groupColumns?: RoleColumn[]
  covariateColumns?: RoleColumn[]
  evaluationColumns?: RoleColumn[]
  ignoredColumns?: RoleColumn[]
}
type StatusWithDataset = WebRunStatus & { datasetUnderstanding?: DatasetUnderstanding; checkpoint?: unknown }

interface ApprovalPanelProps {
  runId?: string
  interaction: WebAgentInteraction
  reasoning?: WebReasoning
  onApproved: () => void
  appearance?: 'panel' | 'conversation'
  decisionError?: string
}

export const ApprovalPanel = ({
  runId,
  interaction,
  reasoning,
  onApproved,
  appearance = 'panel',
  decisionError,
}: ApprovalPanelProps): React.ReactElement | null => {
  const { locale, t } = usePreferences()
  const card = interaction.card
  const [status, setStatus] = useState<StatusWithDataset>()
  const [reason, setReason] = useState('')
  const [busy, setBusy] = useState<'accept' | 'reject' | 'model'>()
  const [error, setError] = useState<string>()

  useEffect(() => {
    if (!runId || (card?.kind !== 'dataset_review' && card?.kind !== 'column_review')) return
    void getStatus(runId).then((value) => setStatus(value as StatusWithDataset)).catch((loadError: unknown) => {
      setError(loadError instanceof Error ? loadError.message : String(loadError))
    })
  }, [runId, card?.kind])

  if (!runId || !card || card.kind === 'research_question' || card.kind === 'dataset_upload') return null

  if (card.kind === 'embedding_choice') return <EmbeddingChoiceCard key={`${runId}:${card.actionRef}:${card.contentHash}`} runId={runId} interaction={interaction} onApproved={onApproved} decisionError={decisionError} />

  if (card.trainingEditable) return <TrainingConfigurationCard key={`${runId}:${card.actionRef}:${card.contentHash}`} runId={runId} interaction={interaction} onApproved={onApproved} />

  if (card.kind === 'action_review') return <ActionApprovalCard key={`${runId}:${card.actionRef}:${card.contentHash}`} runId={runId} interaction={interaction} onApproved={onApproved} decisionError={decisionError} />

  const selectModel = async (modelId: string): Promise<void> => {
    setBusy('model')
    setError(undefined)
    try {
      await postAction(runId, { action: 'selectModel', modelId })
      onApproved()
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : String(submitError))
    } finally {
      setBusy(undefined)
    }
  }

  if (card.kind === 'model_selection') {
    const choices = reasoning?.modelChoices ?? []
    return (
      <section className={`${css.approvalBar} ${css.modelSelectionBar} ${appearance === 'conversation' ? css.approvalConversation : ''}`}>
        <div className={css.approvalBarCopy}>
          {appearance !== 'conversation' && <span>MODEL CHOICE</span>}
          <strong>{card.title}</strong>
          <p>{card.description}</p>
        </div>
        <div className={css.modelChoiceGrid}>
          {choices.map((choice) => (
            <button
              key={choice.modelId}
              type="button"
              className={`${css.modelChoice} ${choice.recommended ? css.modelChoiceRecommended : ''}`}
              disabled={busy != null}
              onClick={() => void selectModel(choice.modelId)}
            >
              <span className={css.modelChoiceHeading}>
                <strong>{choice.modelId.toUpperCase()}</strong>
                {choice.recommended && <small>{locale === 'zh-CN' ? '推荐' : 'Recommended'}</small>}
              </span>
              <p>{choice.rationale}</p>
              <dl>
                <div><dt>{locale === 'zh-CN' ? '优势' : 'Advantages'}</dt><dd>{choice.advantages.join('；')}</dd></div>
                <div><dt>{locale === 'zh-CN' ? '取舍' : 'Tradeoffs'}</dt><dd>{choice.tradeoffs.join('；')}</dd></div>
              </dl>
              <span className={css.modelChoiceAction}>{busy === 'model' ? '…' : (locale === 'zh-CN' ? '选择这个模型' : 'Choose model')}</span>
            </button>
          ))}
        </div>
        {choices.length === 0 && <div className={css.formError}>{locale === 'zh-CN' ? '候选模型仍在生成，请稍后刷新。' : 'Model candidates are still being prepared.'}</div>}
        {error != null && <div className={css.formError}>{error}</div>}
      </section>
    )
  }

  const accept = async (): Promise<void> => {
    setBusy('accept')
    setError(undefined)
    try {
      if (status?.checkpoint != null) {
        await postAction(runId, { action: 'approveCheckpoint' })
      } else if (card.kind === 'dataset_review' || card.kind === 'column_review') {
        const understanding = status?.datasetUnderstanding
        if (!understanding?.textColumns?.length) throw new Error(locale === 'zh-CN' ? '还没有可确认的正文列。' : 'No text column is available for confirmation.')
        await postAction(runId, {
          action: 'confirmDataset',
          status: 'confirmed',
          domainLabel: understanding.domain?.label ?? '通用文本数据',
          analysisUnit: understanding.analysisUnit ?? '每一行是一条独立记录',
          textColumns: columns(understanding.textColumns),
          timeColumns: columns(understanding.timeColumns),
          idColumns: columns(understanding.idColumns),
          metadataColumns: columns(understanding.metadataColumns),
          groupColumns: columns(understanding.groupColumns),
          covariateColumns: columns(understanding.covariateColumns),
          evaluationColumns: columns(understanding.evaluationColumns),
          ignoredColumns: columns(understanding.ignoredColumns),
        })
      } else if (card.kind === 'research_intent_review') {
        await postAction(runId, { action: 'confirmIntent' })
      } else if (card.kind === 'plan_review') {
        await postAction(runId, { action: 'approvePlan', acceptDegradation: false })
      } else if (card.kind === 'training_review') {
        await postAction(runId, { action: 'startTraining' })
      }
      onApproved()
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : String(submitError))
    } finally {
      setBusy(undefined)
    }
  }

  const reject = async (): Promise<void> => {
    setBusy('reject')
    setError(undefined)
    try {
      const feedback = reason.trim() || (locale === 'zh-CN'
        ? '我拒绝当前建议，请重新评估并向我询问需要修正的信息。'
        : 'I reject this proposal. Re-evaluate it and ask what should be corrected.')
      if (card.kind === 'dataset_review') {
        await postAction(runId, { action: 'correctDataset', text: feedback })
      } else if (card.kind === 'column_review') {
        await postAction(runId, { action: 'message', text: feedback, useLanguageProvider: true })
      } else if (card.kind === 'plan_review') {
        await postAction(runId, { action: 'adjustPlan', text: feedback })
      } else {
        await postAction(runId, { action: 'reject', reason: feedback })
      }
      onApproved()
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : String(submitError))
    } finally {
      setBusy(undefined)
    }
  }

  const planSummary = card.kind === 'plan_review' ? reasoning?.plan?.presentation?.summary : undefined
  const parameterConfiguration = card.kind === 'plan_review' ? reasoning?.parameterConfiguration : undefined
  return (
    <section className={`${css.approvalBar} ${appearance === 'conversation' ? css.approvalConversation : ''}`}>
      <div className={css.approvalBarCopy}>
        {appearance !== 'conversation' && <span>HUMAN CHECK</span>}
        <strong>{card.title}</strong>
        <p>{planSummary ?? card.description}</p>
      </div>
      {parameterConfiguration && (
        <div className={css.parameterSummary}>
          <strong>{parameterConfiguration.modelId.toUpperCase()}</strong>
          <div>
            {parameterConfiguration.parameters.map((parameter) => (
              <span key={parameter.field} title={parameter.rationale}>
                <b>{parameter.field}</b>
                <code>{String(parameter.value)}</code>
              </span>
            ))}
          </div>
          <small>{locale === 'zh-CN' ? '在下方直接输入要修改的参数和值，然后选择“调整”。' : 'Enter parameter changes below, then choose Adjust.'}</small>
        </div>
      )}
      <input
        value={reason}
        placeholder={t('reason')}
        aria-label={t('reason')}
        onChange={(event) => setReason(event.target.value)}
      />
      <div className={css.approvalBarActions}>
        <Button size="sm" variant="ghost" disabled={busy != null} onClick={() => void reject()}>
          {busy === 'reject' ? '…' : (card.kind === 'plan_review' ? (locale === 'zh-CN' ? '调整' : 'Adjust') : t('reject'))}
        </Button>
        <Button size="sm" variant="primary" disabled={busy != null} onClick={() => void accept()}>
          {busy === 'accept' ? '…' : t('accept')}
        </Button>
      </div>
      {error != null && <div className={css.formError}>{error}</div>}
    </section>
  )
}

const columns = (items?: RoleColumn[]): string[] => items?.map((item) => item.column) ?? []
