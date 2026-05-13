Name:           quark
Version:        0.1.0
Release:        1%{?dist}
Summary:        Train and run your own Llama 4-style MoE coding LLM

License:        MIT
URL:            https://github.com/0xnullsect0r/Quark

# Pre-built binaries are copied in by build/linux/rpm.sh
# before rpmbuild is invoked — no Source0 needed for binary RPM.
BuildArch:      x86_64

Requires:       gtk3
Requires:       openssl-libs

%description
Quark is a downloadable desktop application that lets you configure,
train from scratch, or fine-tune a Llama 4-style Mixture-of-Experts (MoE)
transformer LLM entirely on your own hardware — no cloud required.

It ships three binaries:
  quark       — the main GUI training + inference application
  quark-chat  — a lightweight terminal REPL for chatting with a trained model
  quark-code  — a full AI coding agent (like Claude Code / GitHub Copilot CLI)
                with project context, slash commands, and MCP tool support

%install
rm -rf %{buildroot}
mkdir -p %{buildroot}%{_bindir}
mkdir -p %{buildroot}%{_datadir}/applications

install -m 755 %{_builddir}/quark       %{buildroot}%{_bindir}/quark
install -m 755 %{_builddir}/quark-chat  %{buildroot}%{_bindir}/quark-chat
install -m 755 %{_builddir}/quark-code  %{buildroot}%{_bindir}/quark-code

install -m 644 %{_builddir}/quark.desktop %{buildroot}%{_datadir}/applications/quark.desktop

%post
if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database -q %{_datadir}/applications || :
fi

%postun
if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database -q %{_datadir}/applications || :
fi

%files
%{_bindir}/quark
%{_bindir}/quark-chat
%{_bindir}/quark-code
%{_datadir}/applications/quark.desktop

%changelog
* Wed May 14 2025 Quark Contributors <https://github.com/0xnullsect0r/Quark> - 0.1.0-1
- Initial release
