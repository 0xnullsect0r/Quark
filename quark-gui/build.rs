fn main() {
    // Embed the nearest git tag version (e.g. "1.0.7") as QUARK_VERSION so
    // the binary always displays the tag it was built from.  Falls back to
    // CARGO_PKG_VERSION if git is unavailable (e.g. inside a tarball build).
    let version = git_tag_version().unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string());
    println!("cargo:rustc-env=QUARK_VERSION={version}");
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/packed-refs");
}

fn git_tag_version() -> Option<String> {
    let out = std::process::Command::new("git")
        .args(["describe", "--tags", "--abbrev=0"])
        .output()
        .ok()?;
    if out.status.success() {
        let tag = String::from_utf8(out.stdout).ok()?;
        Some(tag.trim().trim_start_matches('v').to_string())
    } else {
        None
    }
}
