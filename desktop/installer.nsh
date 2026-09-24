; Reuse the registered installation scope so an upgrade removes both old
; per-user and per-machine installations instead of adding another copy.
!macro customInstallMode
  ${If} $hasPerMachineInstallation == "1"
    StrCpy $isForceMachineInstall "1"
  ${Else}
    StrCpy $isForceCurrentInstall "1"
  ${EndIf}
!macroend

; Keep the dedicated local data folder through upgrades and uninstall. Everything
; else in the program directory remains disposable application payload.
!macro customRemoveFiles
  SetOutPath $TEMP
  Push $R0
  Push $R1
  FindFirst $R0 $R1 "$INSTDIR\*.*"
  theta_remove_next:
    StrCmp $R1 "" theta_remove_done
    StrCmp $R1 "." theta_remove_continue
    StrCmp $R1 ".." theta_remove_continue
    StrCmp $R1 "THETA-data" theta_remove_continue
    IfFileExists "$INSTDIR\$R1\*.*" theta_remove_directory
    ClearErrors
    Delete "$INSTDIR\$R1"
    IfErrors theta_remove_failed
    Goto theta_remove_continue
  theta_remove_directory:
    ClearErrors
    RMDir /r "$INSTDIR\$R1"
    IfErrors theta_remove_failed
  theta_remove_continue:
    FindNext $R0 $R1
    Goto theta_remove_next
  theta_remove_done:
    FindClose $R0
    RMDir "$INSTDIR"
    Pop $R1
    Pop $R0
    Goto theta_remove_success
  theta_remove_failed:
    FindClose $R0
    Abort "Cannot remove application file $INSTDIR\$R1. Close THETA and retry."
  theta_remove_success:
!macroend
