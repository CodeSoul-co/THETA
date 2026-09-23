/**
 * 提示词配置导出
 * 所有硬编码的UI文案、提示语、AI提示词都从 lib/config/prompts.json 中管理
 */

import promt from './prompts.json'

export const PROMPTS = {
  SIDEBAR: promt.sidebar,
  CHAT: promt.chat,
  DASHBOARD: promt.dashboard,
  VISUALIZATION: promt.visualization,
  AUTO_PIPELINE: promt.auto_pipeline,
  BUTTONS: promt.buttons,
}

export default PROMPTS
