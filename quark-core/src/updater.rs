//! Lightweight update checker — queries GitHub Releases API for newer versions.
//! Runs in a background thread; signals the GUI via a channel when an update is available.

use std::sync::mpsc;
use std::time::Duration;

const RELEASES_URL: &str =
    "https://api.github.com/repos/0xnullsect0r/Quark/releases/latest";
const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");
const CHECK_TIMEOUT_SECS: u64 = 8;

#[derive(Debug, Clone)]
pub struct UpdateInfo {
    pub version: String,
    pub html_url: String,
    pub body: String,
}

/// Spawn a background thread that checks for updates and sends the result once.
/// Returns a receiver; the caller polls `try_recv()` on each frame.
pub fn spawn_update_check() -> mpsc::Receiver<Option<UpdateInfo>> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let result = check_for_update();
        let _ = tx.send(result);
    });
    rx
}

fn check_for_update() -> Option<UpdateInfo> {
    let resp = ureq::builder()
        .timeout(Duration::from_secs(CHECK_TIMEOUT_SECS))
        .build()
        .get(RELEASES_URL)
        .set("User-Agent", &format!("quark/{CURRENT_VERSION}"))
        .call()
        .ok()?;

    let json: serde_json::Value = resp.into_json().ok()?;
    let tag = json["tag_name"].as_str()?;
    let version = tag.trim_start_matches('v').to_string();

    if semver_newer(&version, CURRENT_VERSION) {
        Some(UpdateInfo {
            version,
            html_url: json["html_url"].as_str().unwrap_or("").to_string(),
            body: json["body"].as_str().unwrap_or("").chars().take(400).collect(),
        })
    } else {
        None
    }
}

/// Returns true if `candidate` is strictly newer than `current` (simple semver compare).
fn semver_newer(candidate: &str, current: &str) -> bool {
    parse_semver(candidate) > parse_semver(current)
}

fn parse_semver(v: &str) -> (u32, u32, u32) {
    let parts: Vec<u32> = v.split('.').filter_map(|s| s.parse().ok()).collect();
    (
        parts.first().copied().unwrap_or(0),
        parts.get(1).copied().unwrap_or(0),
        parts.get(2).copied().unwrap_or(0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_semver_newer() {
        assert!(semver_newer("0.2.0", "0.1.0"));
        assert!(semver_newer("1.0.0", "0.9.9"));
        assert!(!semver_newer("0.1.0", "0.1.0"));
        assert!(!semver_newer("0.0.9", "0.1.0"));
    }
}
