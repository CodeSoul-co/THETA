; Reuse the registered installation scope so an upgrade removes both old
; per-user and per-machine installations instead of adding another copy.
!macro customInstallMode
  ${If} $hasPerMachineInstallation == "1"
    StrCpy $isForceMachineInstall "1"
  ${Else}
    StrCpy $isForceCurrentInstall "1"
  ${EndIf}
!macroend
