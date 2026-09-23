import type { IconProps } from './icons/props.ts'
import { ThetaCubeMark } from './ThetaCubeMark.tsx'
import css from './BrandWordmark.module.css'

/** Render the THETA Agent product mark used by the application shell. */
export function BrandWordmark({ size = 24, className }: IconProps) {
  return (
    <div className={`${css.mark} ${className ?? ''}`} style={{ height: size }} aria-label="THETA Agent">
      <ThetaCubeMark size={34} />
      <span className={css.name}>THETA</span>
    </div>
  )
}
