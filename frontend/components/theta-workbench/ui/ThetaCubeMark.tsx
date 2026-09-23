import type { IconProps } from './icons/props.ts'

export const ThetaCubeMark = ({ size = 32, className }: IconProps): React.ReactElement => (
  <svg
    className={className}
    width={size}
    height={size}
    viewBox="0 0 64 64"
    fill="none"
    stroke="currentColor"
    strokeWidth="2.2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <path d="M32 4 58 18v28L32 60 6 46V18L32 4Z" />
    <path d="m6 18 26 14 26-14M32 32v28M32 4v28" />
    <path d="m32 16 14 8v16l-14 8-14-8V24l14-8Z" />
    <path d="m18 24 14 8 14-8M18 40l14-8 14 8" />
  </svg>
)
