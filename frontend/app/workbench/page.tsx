'use client'

import dynamic from 'next/dynamic'

const ThetaWorkbench = dynamic(
  () => import('@/components/theta-workbench/ThetaWorkbench').then((module) => module.ThetaWorkbench),
  { ssr: false },
)

export default function WorkbenchPage() {
  return <ThetaWorkbench />
}
