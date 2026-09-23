import type { ImgHTMLAttributes } from 'react'

export function CatBrandWordmark(props: Omit<ImgHTMLAttributes<HTMLImageElement>, 'src' | 'alt'>) {
  return (
    <img
      {...props}
      src="/theta-assets/brand/theta-cat-wordmark-final.png"
      alt="THETA"
      draggable={false}
    />
  )
}
