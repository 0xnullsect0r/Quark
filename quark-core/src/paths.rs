//! Platform-specific application directory helpers.
//!
//! | Platform | Path                                      |
//! |----------|-------------------------------------------|
//! | Windows  | `%APPDATA%\Quark`                         |
//! | macOS    | `~/Library/Application Support/Quark`     |
//! | Linux    | `~/.quark`                                |

use std::path::PathBuf;

/// Returns the root application data directory for Quark.
///
/// All persistent user data (datasets, checkpoints, settings) lives under
/// this directory.
pub fn app_data_dir() -> PathBuf {
    #[cfg(target_os = "linux")]
    {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".quark")
    }
    #[cfg(target_os = "macos")]
    {
        dirs::data_dir() // ~/Library/Application Support
            .unwrap_or_else(|| PathBuf::from("."))
            .join("Quark")
    }
    #[cfg(target_os = "windows")]
    {
        dirs::data_dir() // %APPDATA% (Roaming)
            .unwrap_or_else(|| PathBuf::from("."))
            .join("Quark")
    }
    // Fallback for any other platform
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        dirs::data_dir()
            .or_else(dirs::home_dir)
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".quark")
    }
}

/// Convenience: `<app_data_dir>/settings.toml`
pub fn settings_path() -> PathBuf {
    app_data_dir().join("settings.toml")
}

/// Convenience: `<app_data_dir>/the-pile` root for Pile dataset builds.
pub fn pile_data_dir() -> PathBuf {
    app_data_dir().join("the-pile")
}

/// Convenience: `<app_data_dir>/checkpoints`
pub fn checkpoints_dir() -> PathBuf {
    app_data_dir().join("checkpoints")
}

/// Convenience: `<app_data_dir>/datasets` — where downloaded JSONL files live.
pub fn datasets_dir() -> PathBuf {
    app_data_dir().join("datasets")
}
