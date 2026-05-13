; EnvVarUpdate.nsh — NSIS helper for adding/removing PATH entries
; Sourced from the NSIS wiki (public domain).
; ${EnvVarUpdate} $0 "PATH" "A" "HKLM" "C:\new\path"   ; Add
; ${EnvVarUpdate} $0 "PATH" "R" "HKLM" "C:\old\path"   ; Remove

!ifndef ENVVARUPDATE_NSH
!define ENVVARUPDATE_NSH

!include "LogicLib.nsh"
!include "WinMessages.nsh"
!include "StrFunc.nsh"
${StrStr}
${UnStrStr}
${StrStrAdv}
${UnStrStrAdv}
${StrRep}
${UnStrRep}

!macro _EnvVarUpdateConstructor ResultVar EnvVarName Action RegLoc PathString
  Push "${EnvVarName}"
  Push "${Action}"
  Push "${RegLoc}"
  Push "${PathString}"
  Call EnvVarUpdate
  Pop "${ResultVar}"
!macroend
!define EnvVarUpdate '!insertmacro _EnvVarUpdateConstructor'

!macro _un.EnvVarUpdateConstructor ResultVar EnvVarName Action RegLoc PathString
  Push "${EnvVarName}"
  Push "${Action}"
  Push "${RegLoc}"
  Push "${PathString}"
  Call un.EnvVarUpdate
  Pop "${ResultVar}"
!macroend
!define un.EnvVarUpdate '!insertmacro _un.EnvVarUpdateConstructor'

!macro EnvVarUpdateBody UN
Function ${UN}EnvVarUpdate
  ; Stack: EnvVarName | Action | RegLoc | PathString
  Exch $0  ; PathString
  Exch
  Exch $1  ; RegLoc
  Exch 2
  Exch $2  ; Action
  Exch 3
  Exch $3  ; EnvVarName
  Push $4
  Push $5
  Push $6
  Push $7
  Push $8

  ReadRegStr $4 ${$1} "Environment" $3
  ${If} $4 == ""
    StrCpy $4 $0
    ${If} $2 == "A"
      WriteRegExpandStr ${$1} "Environment" $3 $4
    ${EndIf}
    Goto EVU_end
  ${EndIf}

  ; Check if value already present
  ${StrStr} $5 $4 $0
  ${If} $2 == "A"
    ${If} $5 != ""
      Goto EVU_end  ; already there
    ${EndIf}
    StrCpy $4 "$4;$0"
    WriteRegExpandStr ${$1} "Environment" $3 $4
  ${Else}  ; Remove
    ${If} $5 == ""
      Goto EVU_end  ; not found
    ${EndIf}
    ; Strip all occurrences (handles leading, trailing, middle)
    ${StrRep} $4 $4 ";$0" ""
    ${StrRep} $4 $4 "$0;" ""
    ${StrRep} $4 $4 "$0"  ""
    WriteRegExpandStr ${$1} "Environment" $3 $4
  ${EndIf}

  EVU_end:
  Pop $8
  Pop $7
  Pop $6
  Pop $5
  Pop $4
  Pop $3
  Pop $2
  Pop $1
  Exch $0
FunctionEnd
!macroend

!insertmacro EnvVarUpdateBody ""
!insertmacro EnvVarUpdateBody "un."

!endif ; ENVVARUPDATE_NSH
