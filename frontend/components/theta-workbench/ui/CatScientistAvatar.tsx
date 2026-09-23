import css from './CatScientistAvatar.module.css'

interface CatScientistAvatarProps {
  size?: number
  className?: string
}

export const CatScientistAvatar = ({ size = 24, className }: CatScientistAvatarProps): React.ReactElement => (
  <img
    className={`${css.avatar} ${className ?? ''}`}
    src="/brand/theta-cat-scientist.png"
    alt=""
    aria-hidden="true"
    width={size}
    height={size}
  />
)
