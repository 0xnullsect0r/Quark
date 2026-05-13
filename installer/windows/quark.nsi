; Quark LLM — Windows NSIS Installer
; Requires NSIS 3.x  (https://nsis.sourceforge.io)
; Build with: makensis installer/windows/quark.nsi

!define APP_NAME      "Quark"
!define APP_VERSION   "0.1.0"
!define APP_PUBLISHER "Quark Contributors"
!define APP_URL       "https://github.com/0xnullsect0r/Quark"
!define APP_EXE       "quark.exe"
!define INSTALL_DIR   "$PROGRAMFILES64\Quark"

Name "${APP_NAME} ${APP_VERSION}"
OutFile "Quark-${APP_VERSION}-windows-setup.exe"
InstallDir "${INSTALL_DIR}"
InstallDirRegKey HKLM "Software\${APP_NAME}" ""
RequestExecutionLevel admin
SetCompressor /SOLID lzma

!include "MUI2.nsh"

!define MUI_ABORTWARNING

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Section "Quark (required)" SecCore
  SectionIn RO
  SetOutPath "$INSTDIR"
  File "..\..\target\release\${APP_EXE}"
  WriteRegStr HKLM "Software\${APP_NAME}" "" "$INSTDIR"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
    "DisplayName" "${APP_NAME} ${APP_VERSION}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
    "UninstallString" "$INSTDIR\uninstall.exe"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
    "DisplayVersion" "${APP_VERSION}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
    "Publisher" "${APP_PUBLISHER}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
    "URLInfoAbout" "${APP_URL}"
  WriteUninstaller "$INSTDIR\uninstall.exe"
SectionEnd

Section "Start Menu Shortcuts"
  CreateDirectory "$SMPROGRAMS\${APP_NAME}"
  CreateShortcut  "$SMPROGRAMS\${APP_NAME}\Quark.lnk" "$INSTDIR\${APP_EXE}"
  CreateShortcut  "$SMPROGRAMS\${APP_NAME}\Uninstall Quark.lnk" "$INSTDIR\uninstall.exe"
  CreateShortcut  "$DESKTOP\Quark.lnk" "$INSTDIR\${APP_EXE}"
SectionEnd

Section "Uninstall"
  Delete "$INSTDIR\${APP_EXE}"
  Delete "$INSTDIR\uninstall.exe"
  RMDir  "$INSTDIR"
  Delete "$SMPROGRAMS\${APP_NAME}\Quark.lnk"
  Delete "$SMPROGRAMS\${APP_NAME}\Uninstall Quark.lnk"
  Delete "$DESKTOP\Quark.lnk"
  RMDir  "$SMPROGRAMS\${APP_NAME}"
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}"
  DeleteRegKey HKLM "Software\${APP_NAME}"
SectionEnd
