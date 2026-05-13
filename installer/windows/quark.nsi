; Quark LLM — Windows NSIS Installer
; Requires NSIS 3.x  (https://nsis.sourceforge.io)
; Build with: makensis installer/windows/quark.nsi

!define APP_NAME      "Quark"
!define APP_VERSION   "0.1.0"
!define APP_PUBLISHER "Quark Contributors"
!define APP_URL       "https://github.com/0xnullsect0r/Quark"
!define APP_EXE       "quark.exe"
!define APP_CHAT_EXE  "quark-chat.exe"
!define APP_CODE_EXE  "quark-code.exe"
!define INSTALL_DIR   "$PROGRAMFILES64\Quark"
!define UNINST_KEY    "Software\Microsoft\Windows\CurrentVersion\Uninstall\Quark"

Name "${APP_NAME} ${APP_VERSION}"
OutFile "Quark-${APP_VERSION}-windows-setup.exe"
InstallDir "${INSTALL_DIR}"
InstallDirRegKey HKLM "Software\${APP_NAME}" "InstallDir"
RequestExecutionLevel admin
SetCompressor /SOLID lzma
Unicode True

;--------------------------------
; MUI2 Interface
;--------------------------------
!include "MUI2.nsh"
!include "EnvVarUpdate.nsh"
!include "Sections.nsh"

!define MUI_ABORTWARNING
!define MUI_WELCOMEPAGE_TITLE  "Welcome to the Quark ${APP_VERSION} Setup"
!define MUI_WELCOMEPAGE_TEXT   "Quark lets you train and run your own Llama 4-style MoE coding LLM entirely on your own hardware — no cloud required.$\r$\n$\r$\nThis wizard will guide you through the installation."
!define MUI_FINISHPAGE_RUN           "$INSTDIR\${APP_EXE}"
!define MUI_FINISHPAGE_RUN_TEXT      "Launch Quark"
!define MUI_FINISHPAGE_SHOWREADME    "$INSTDIR\README.txt"
!define MUI_FINISHPAGE_SHOWREADME_TEXT "View README"

; Pages
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "..\..\LICENSE"
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

;--------------------------------
; Section Descriptions
;--------------------------------
LangString DESC_SecCore   ${LANG_ENGLISH} "Quark GUI — the main LLM training and inference application. Required."
LangString DESC_SecChat   ${LANG_ENGLISH} "quark-chat — lightweight terminal REPL for chatting with a trained model."
LangString DESC_SecCode   ${LANG_ENGLISH} "quark-code — AI coding agent (like Claude Code). Adds quark-code to your PATH."
LangString DESC_SecShortcuts ${LANG_ENGLISH} "Create Start Menu and Desktop shortcuts for Quark."

;--------------------------------
; Core (required)
;--------------------------------
Section "Quark GUI (required)" SecCore
  SectionIn RO
  SetOutPath "$INSTDIR"

  File "..\..\target\release\${APP_EXE}"
  File "..\..\LICENSE"
  ; Write a plain-text README stub
  FileOpen $0 "$INSTDIR\README.txt" w
  FileWrite $0 "Quark ${APP_VERSION}$\r$\nProject: ${APP_URL}$\r$\n"
  FileClose $0

  ; Registry: install location
  WriteRegStr HKLM "Software\${APP_NAME}" "InstallDir" "$INSTDIR"
  WriteRegStr HKLM "Software\${APP_NAME}" "Version"    "${APP_VERSION}"

  ; Add/Remove Programs entry
  WriteRegStr HKLM "${UNINST_KEY}" "DisplayName"    "${APP_NAME} ${APP_VERSION}"
  WriteRegStr HKLM "${UNINST_KEY}" "DisplayVersion"  "${APP_VERSION}"
  WriteRegStr HKLM "${UNINST_KEY}" "Publisher"       "${APP_PUBLISHER}"
  WriteRegStr HKLM "${UNINST_KEY}" "URLInfoAbout"    "${APP_URL}"
  WriteRegStr HKLM "${UNINST_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKLM "${UNINST_KEY}" "UninstallString" "$INSTDIR\uninstall.exe"
  WriteRegStr HKLM "${UNINST_KEY}" "DisplayIcon"     "$INSTDIR\${APP_EXE}"
  WriteRegDWORD HKLM "${UNINST_KEY}" "NoModify"      1
  WriteRegDWORD HKLM "${UNINST_KEY}" "NoRepair"      1

  WriteUninstaller "$INSTDIR\uninstall.exe"
SectionEnd

;--------------------------------
; quark-chat (optional)
;--------------------------------
Section "Quark Chat (terminal REPL)" SecChat
  SetOutPath "$INSTDIR"
  File "..\..\target\release\${APP_CHAT_EXE}"
SectionEnd

;--------------------------------
; quark-code (optional, adds to PATH)
;--------------------------------
Section "Quark Code (AI coding agent + PATH)" SecCode
  SetOutPath "$INSTDIR"
  File "..\..\target\release\${APP_CODE_EXE}"
  ; Add install dir to system PATH (idempotent via EnvVarUpdate)
  ${EnvVarUpdate} $0 "PATH" "A" "HKLM" "$INSTDIR"
  SendMessage ${HWND_BROADCAST} ${WM_WININICHANGE} 0 "STR:Environment" /TIMEOUT=5000
SectionEnd

;--------------------------------
; Start Menu / Desktop shortcuts
;--------------------------------
Section "Shortcuts" SecShortcuts
  CreateDirectory "$SMPROGRAMS\${APP_NAME}"
  CreateShortcut  "$SMPROGRAMS\${APP_NAME}\Quark.lnk"            "$INSTDIR\${APP_EXE}"
  CreateShortcut  "$SMPROGRAMS\${APP_NAME}\Uninstall Quark.lnk"  "$INSTDIR\uninstall.exe"
  CreateShortcut  "$DESKTOP\Quark.lnk"                            "$INSTDIR\${APP_EXE}"
SectionEnd

; Attach descriptions to the component page
!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecCore}      $(DESC_SecCore)
  !insertmacro MUI_DESCRIPTION_TEXT ${SecChat}      $(DESC_SecChat)
  !insertmacro MUI_DESCRIPTION_TEXT ${SecCode}      $(DESC_SecCode)
  !insertmacro MUI_DESCRIPTION_TEXT ${SecShortcuts} $(DESC_SecShortcuts)
!insertmacro MUI_FUNCTION_DESCRIPTION_END

;--------------------------------
; Uninstaller
;--------------------------------
Section "Uninstall"
  ; Remove PATH entry added by SecCode
  ${un.EnvVarUpdate} $0 "PATH" "R" "HKLM" "$INSTDIR"
  SendMessage ${HWND_BROADCAST} ${WM_WININICHANGE} 0 "STR:Environment" /TIMEOUT=5000

  Delete "$INSTDIR\${APP_EXE}"
  Delete "$INSTDIR\${APP_CHAT_EXE}"
  Delete "$INSTDIR\${APP_CODE_EXE}"
  Delete "$INSTDIR\LICENSE"
  Delete "$INSTDIR\README.txt"
  Delete "$INSTDIR\uninstall.exe"
  RMDir  "$INSTDIR"

  Delete "$SMPROGRAMS\${APP_NAME}\Quark.lnk"
  Delete "$SMPROGRAMS\${APP_NAME}\Uninstall Quark.lnk"
  Delete "$DESKTOP\Quark.lnk"
  RMDir  "$SMPROGRAMS\${APP_NAME}"

  DeleteRegKey HKLM "${UNINST_KEY}"
  DeleteRegKey HKLM "Software\${APP_NAME}"
SectionEnd
