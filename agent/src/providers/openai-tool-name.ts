export const openAiToolFunctionName = (toolId: string): string => {
  const normalized = toolId.replace(/[^A-Za-z0-9_-]/gu, '_');
  if (!normalized) throw new Error(`Tool id cannot be represented as a function name: ${toolId}`);
  return normalized.slice(0, 64);
};
