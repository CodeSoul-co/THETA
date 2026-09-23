import type { ProductSession } from './session-store.js';

/** Count human input, not tool receipts, background monitor prompts or card clicks. */
export function userMessageCount(session: ProductSession): number {
  return session.messages.filter(message => message.role === 'user' &&
    message.metadata?.webHostEvent !== true && message.metadata?.webAction !== true &&
    // Compatibility with old training-dialog receipts, before webAction existed.
    !(message.metadata?.webUserText === undefined && /^确认训练配置：/u.test(message.content))).length;
}
