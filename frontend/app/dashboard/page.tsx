"use client"

import dynamic from "next/dynamic"

const ThetaWorkbench = dynamic(
  () => import("@/components/theta-workbench/ThetaWorkbench").then((module) => module.ThetaWorkbench),
  { ssr: false },
)

// The legacy URL opens the same workbench shell, initially in manual mode.
export default function DashboardPage() {
  return <ThetaWorkbench initialMode="manual" />
}
