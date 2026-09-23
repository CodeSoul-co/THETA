export const THETA_PASSWORD_POLICY_MESSAGE = "密码为 6-20 位，仅可包含英文字母、数字和 @，且必须同时包含英文字母和数字"

export const isThetaPasswordValid = (value: string): boolean => (
  /^[A-Za-z0-9@]{6,20}$/u.test(value)
  && /[A-Za-z]/u.test(value)
  && /\d/u.test(value)
)

export const normalizeThetaPasswordInput = (value: string): string => (
  value.replace(/[^A-Za-z0-9@]/gu, "").slice(0, 20)
)
