//! Tiered host-side tensor store: an LRU RAM cache over per-stage
//! safetensors files on disk.
//!
//! Training and inference for models larger than memory work one **stage** at
//! a time (the embedding, each decoder layer, each expert, the head, and the
//! optimizer state for each). A stage is a list of named tensors. The store
//! keeps recently used stages in RAM up to `ram_limit` bytes, writes changed
//! stages back to disk when they are evicted, and can load stages on a
//! background thread ahead of use ([`TensorStore::prefetch`]).
//!
//! Device (VRAM) residency is handled by the caller: it turns a stage into a
//! Burn module on the compute device and drops it when done.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use anyhow::{Context, Result};
use burn::tensor::{DType, TensorData};
use safetensors::tensor::{Dtype, SafeTensors, TensorView};

/// The named tensors of one stage, in a stable order.
#[derive(Debug, Clone, Default)]
pub struct StageTensors {
    pub tensors: Vec<(String, TensorData)>,
}

impl StageTensors {
    pub fn new(tensors: Vec<(String, TensorData)>) -> Self {
        Self { tensors }
    }

    pub fn get(&self, name: &str) -> Option<&TensorData> {
        self.tensors.iter().find(|(n, _)| n == name).map(|(_, t)| t)
    }

    /// Size of the tensor payloads in bytes.
    pub fn bytes(&self) -> u64 {
        self.tensors.iter().map(|(_, t)| t.as_bytes().len() as u64).sum()
    }
}

/// Bytes moved between the store and disk since it was created.
#[derive(Debug, Default)]
pub struct IoStats {
    pub read_bytes: AtomicU64,
    pub write_bytes: AtomicU64,
}

#[derive(Debug)]
struct Entry {
    data: Arc<StageTensors>,
    bytes: u64,
    dirty: bool,
    /// Version of `data` (bumped by every `put`).
    version: u64,
}

/// A stage copy being written to disk; still served from here until done.
struct Pending {
    data: Arc<StageTensors>,
    version: u64,
}

#[derive(Default)]
struct Inner {
    cache: HashMap<String, Entry>,
    /// Least recently used at the front.
    lru: VecDeque<String>,
    ram_used: u64,
    writing: HashMap<String, Pending>,
    /// Stages being loaded (by `get` or `prefetch`).
    loading: HashMap<String, ()>,
    /// Latest version per stage, and the version on disk.
    latest: HashMap<String, u64>,
    on_disk: HashMap<String, u64>,
}

/// See the module docs.
pub struct TensorStore {
    dir: PathBuf,
    ram_limit: u64,
    inner: Mutex<Inner>,
    /// Signalled when a load or a write-back finishes.
    changed: Condvar,
    /// Serialises disk writes so an older version never lands after a newer one.
    write_lock: Mutex<()>,
    stats: IoStats,
}

type Evicted = Vec<(String, Arc<StageTensors>, u64)>;

impl TensorStore {
    /// A store keeping at most `ram_limit` bytes of stages in RAM, spilling to
    /// files under `dir`.
    pub fn new(dir: impl Into<PathBuf>, ram_limit: u64) -> Result<Arc<Self>> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("cannot create offload dir {}", dir.display()))?;
        Ok(Arc::new(Self {
            dir,
            ram_limit,
            inner: Mutex::new(Inner::default()),
            changed: Condvar::new(),
            write_lock: Mutex::new(()),
            stats: IoStats::default(),
        }))
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn stats(&self) -> &IoStats {
        &self.stats
    }

    /// Bytes of stages currently held in RAM.
    pub fn ram_used(&self) -> u64 {
        self.inner.lock().unwrap().ram_used
    }

    fn path(&self, stage: &str) -> PathBuf {
        self.dir.join(format!("{stage}.safetensors"))
    }

    /// Whether the stage exists in RAM or on disk.
    pub fn contains(&self, stage: &str) -> bool {
        let inner = self.inner.lock().unwrap();
        inner.cache.contains_key(stage)
            || inner.writing.contains_key(stage)
            || self.path(stage).exists()
    }

    /// Store (or replace) a stage. It stays in RAM until evicted, and is
    /// written to disk then (or on [`Self::flush`]).
    pub fn put(&self, stage: &str, data: StageTensors) -> Result<()> {
        let bytes = data.bytes();
        let evicted = {
            let mut inner = self.inner.lock().unwrap();
            let version = {
                let v = inner.latest.entry(stage.to_owned()).or_insert(0);
                *v += 1;
                *v
            };
            if let Some(old) = inner.cache.remove(stage) {
                inner.ram_used -= old.bytes;
                inner.lru.retain(|s| s != stage);
            }
            let entry = Entry { data: Arc::new(data), bytes, dirty: true, version };
            inner.cache.insert(stage.to_owned(), entry);
            inner.lru.push_back(stage.to_owned());
            inner.ram_used += bytes;
            self.evict_over_limit(&mut inner, stage)
        };
        self.write_all(evicted)
    }

    /// Get a stage, loading it from disk if needed (or waiting for an
    /// in-flight load).
    pub fn get(&self, stage: &str) -> Result<Arc<StageTensors>> {
        {
            let mut inner = self.inner.lock().unwrap();
            loop {
                if let Some(data) = self.touch(&mut inner, stage) {
                    return Ok(data);
                }
                if let Some(p) = inner.writing.get(stage) {
                    return Ok(Arc::clone(&p.data));
                }
                if inner.loading.contains_key(stage) {
                    inner = self.changed.wait(inner).unwrap();
                    continue;
                }
                inner.loading.insert(stage.to_owned(), ());
                break;
            }
        }
        let result = self.read_file(stage);
        self.finish_load(stage, result)
    }

    /// Start loading a stage on a background thread, so a later
    /// [`Self::get`] finds it in RAM.
    pub fn prefetch(self: &Arc<Self>, stage: &str) {
        {
            let mut inner = self.inner.lock().unwrap();
            if inner.cache.contains_key(stage)
                || inner.writing.contains_key(stage)
                || inner.loading.contains_key(stage)
                || !self.path(stage).exists()
            {
                return;
            }
            inner.loading.insert(stage.to_owned(), ());
        }
        let store = Arc::clone(self);
        let stage = stage.to_owned();
        std::thread::spawn(move || {
            let result = store.read_file(&stage);
            // A failed prefetch is retried (and reported) by the next `get`.
            let _ = store.finish_load(&stage, result);
        });
    }

    /// Drop a stage from RAM and disk.
    pub fn remove(&self, stage: &str) -> Result<()> {
        {
            let mut inner = self.inner.lock().unwrap();
            if let Some(old) = inner.cache.remove(stage) {
                inner.ram_used -= old.bytes;
            }
            inner.lru.retain(|s| s != stage);
            inner.on_disk.remove(stage);
            *inner.latest.entry(stage.to_owned()).or_insert(0) += 1;
        }
        match std::fs::remove_file(self.path(stage)) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e.into()),
            _ => Ok(()),
        }
    }

    /// Write every changed stage to disk (keeping it in RAM), and wait for
    /// write-backs started elsewhere.
    pub fn flush(&self) -> Result<()> {
        let dirty: Evicted = {
            let mut inner = self.inner.lock().unwrap();
            let dirty: Evicted = inner
                .cache
                .iter_mut()
                .filter(|(_, e)| e.dirty)
                .map(|(name, e)| {
                    e.dirty = false;
                    (name.clone(), Arc::clone(&e.data), e.version)
                })
                .collect();
            for (stage, data, version) in &dirty {
                inner
                    .writing
                    .insert(stage.clone(), Pending { data: Arc::clone(data), version: *version });
            }
            dirty
        };
        self.write_all(dirty)?;
        let mut inner = self.inner.lock().unwrap();
        while !inner.writing.is_empty() {
            inner = self.changed.wait(inner).unwrap();
        }
        Ok(())
    }

    /// Flush, then copy every stage file whose name starts with `prefix` into
    /// `dest` (e.g. a checkpoint directory).
    pub fn copy_stages_to(&self, prefix: &str, dest: &Path) -> Result<()> {
        self.flush()?;
        std::fs::create_dir_all(dest)?;
        for entry in std::fs::read_dir(&self.dir)? {
            let path = entry?.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else { continue };
            if name.starts_with(prefix) && name.ends_with(".safetensors") {
                std::fs::copy(&path, dest.join(name))?;
            }
        }
        Ok(())
    }

    // ── internals ────────────────────────────────────────────────────────────

    fn touch(&self, inner: &mut Inner, stage: &str) -> Option<Arc<StageTensors>> {
        let data = Arc::clone(&inner.cache.get(stage)?.data);
        inner.lru.retain(|s| s != stage);
        inner.lru.push_back(stage.to_owned());
        Some(data)
    }

    fn finish_load(&self, stage: &str, result: Result<StageTensors>) -> Result<Arc<StageTensors>> {
        let mut inner = self.inner.lock().unwrap();
        inner.loading.remove(stage);
        self.changed.notify_all();
        let data = Arc::new(result?);
        // A `put` may have raced ahead of this load; keep the newer data.
        if let Some(existing) = self.touch(&mut inner, stage) {
            return Ok(existing);
        }
        let bytes = data.bytes();
        let version = inner.on_disk.get(stage).copied().unwrap_or(0);
        let entry = Entry { data: Arc::clone(&data), bytes, dirty: false, version };
        inner.cache.insert(stage.to_owned(), entry);
        inner.lru.push_back(stage.to_owned());
        inner.ram_used += bytes;
        let evicted = self.evict_over_limit(&mut inner, stage);
        drop(inner);
        self.write_all(evicted)?;
        Ok(data)
    }

    /// Evict least recently used stages (never `keep`) until under the RAM
    /// limit. Returns the dirty ones, which the caller writes out.
    fn evict_over_limit(&self, inner: &mut Inner, keep: &str) -> Evicted {
        let mut dirty = Vec::new();
        let mut i = 0;
        while inner.ram_used > self.ram_limit && i < inner.lru.len() {
            if inner.lru[i] == keep {
                i += 1;
                continue;
            }
            let stage = inner.lru.remove(i).unwrap();
            let entry = inner.cache.remove(&stage).unwrap();
            inner.ram_used -= entry.bytes;
            if entry.dirty {
                let pending = Pending { data: Arc::clone(&entry.data), version: entry.version };
                inner.writing.insert(stage.clone(), pending);
                dirty.push((stage, entry.data, entry.version));
            }
        }
        dirty
    }

    /// Write stages queued in `writing`, then drop them from `writing`.
    fn write_all(&self, stages: Evicted) -> Result<()> {
        let mut result = Ok(());
        for (stage, data, version) in stages {
            let written = self.write_versioned(&stage, &data, version);
            let mut inner = self.inner.lock().unwrap();
            if inner.writing.get(&stage).is_some_and(|p| p.version == version) {
                inner.writing.remove(&stage);
            }
            drop(inner);
            self.changed.notify_all();
            if result.is_ok() {
                result = written;
            }
        }
        result
    }

    /// Write `data` unless a newer version of the stage is already on disk.
    fn write_versioned(&self, stage: &str, data: &StageTensors, version: u64) -> Result<()> {
        let _serial = self.write_lock.lock().unwrap();
        if self.inner.lock().unwrap().on_disk.get(stage).is_some_and(|&v| v >= version) {
            return Ok(());
        }
        let bytes = serialize(data)?;
        let path = self.path(stage);
        let tmp = path.with_extension("safetensors.tmp");
        std::fs::write(&tmp, &bytes).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, &path)?;
        self.stats.write_bytes.fetch_add(bytes.len() as u64, Ordering::Relaxed);
        self.inner.lock().unwrap().on_disk.insert(stage.to_owned(), version);
        Ok(())
    }

    fn read_file(&self, stage: &str) -> Result<StageTensors> {
        let path = self.path(stage);
        let bytes =
            std::fs::read(&path).with_context(|| format!("reading stage {}", path.display()))?;
        self.stats.read_bytes.fetch_add(bytes.len() as u64, Ordering::Relaxed);
        deserialize(&bytes)
    }
}

// ── safetensors encoding ─────────────────────────────────────────────────────

fn to_st_dtype(dtype: DType) -> Result<Dtype> {
    Ok(match dtype {
        DType::F32 => Dtype::F32,
        DType::F64 => Dtype::F64,
        DType::F16 => Dtype::F16,
        DType::BF16 => Dtype::BF16,
        DType::I64 => Dtype::I64,
        DType::I32 => Dtype::I32,
        DType::I16 => Dtype::I16,
        DType::I8 => Dtype::I8,
        DType::U64 => Dtype::U64,
        DType::U32 => Dtype::U32,
        DType::U16 => Dtype::U16,
        DType::U8 => Dtype::U8,
        other => anyhow::bail!("dtype {other:?} cannot be stored"),
    })
}

fn from_st_dtype(dtype: Dtype) -> Result<DType> {
    Ok(match dtype {
        Dtype::F32 => DType::F32,
        Dtype::F64 => DType::F64,
        Dtype::F16 => DType::F16,
        Dtype::BF16 => DType::BF16,
        Dtype::I64 => DType::I64,
        Dtype::I32 => DType::I32,
        Dtype::I16 => DType::I16,
        Dtype::I8 => DType::I8,
        Dtype::U64 => DType::U64,
        Dtype::U32 => DType::U32,
        Dtype::U16 => DType::U16,
        Dtype::U8 => DType::U8,
        other => anyhow::bail!("unsupported safetensors dtype {other:?}"),
    })
}

/// Encode a stage as a safetensors file (tensor order is preserved via
/// `__order__` metadata).
pub fn serialize(data: &StageTensors) -> Result<Vec<u8>> {
    let views = data
        .tensors
        .iter()
        .map(|(name, t)| {
            let view = TensorView::new(to_st_dtype(t.dtype)?, t.shape.to_vec(), t.as_bytes())?;
            Ok((name.as_str(), view))
        })
        .collect::<Result<Vec<_>>>()?;
    let order: Vec<&str> = data.tensors.iter().map(|(n, _)| n.as_str()).collect();
    let metadata = HashMap::from([("__order__".to_owned(), order.join(","))]);
    Ok(safetensors::tensor::serialize(views, &Some(metadata))?)
}

/// Decode a safetensors file written by [`serialize`] (or any safetensors
/// file; without order metadata tensors are sorted by name).
pub fn deserialize(bytes: &[u8]) -> Result<StageTensors> {
    let st = SafeTensors::deserialize(bytes)?;
    let (_, meta) = SafeTensors::read_metadata(bytes)?;
    let mut names: Vec<String> = match meta.metadata().as_ref().and_then(|m| m.get("__order__")) {
        Some(order) if !order.is_empty() => order.split(',').map(str::to_owned).collect(),
        _ => {
            let mut n: Vec<String> = st.names().into_iter().cloned().collect();
            n.sort();
            n
        }
    };
    names.retain(|n| !n.is_empty());
    let tensors = names
        .into_iter()
        .map(|name| {
            let view = st.tensor(&name)?;
            let data = TensorData::from_bytes_vec(
                view.data().to_vec(),
                view.shape().to_vec(),
                from_st_dtype(view.dtype())?,
            );
            Ok((name, data))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(StageTensors { tensors })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stage(fill: f32, n: usize) -> StageTensors {
        StageTensors::new(vec![
            ("w".into(), TensorData::new(vec![fill; n], [n])),
            ("ids".into(), TensorData::new(vec![1u8, 2, 3], [3])),
        ])
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("quark-store-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn roundtrip_preserves_order_and_dtypes() {
        let data = stage(1.5, 4);
        let back = deserialize(&serialize(&data).unwrap()).unwrap();
        assert_eq!(back.tensors[0].0, "w");
        assert_eq!(back.tensors[1].0, "ids");
        assert_eq!(back.get("w").unwrap().to_vec::<f32>().unwrap(), vec![1.5; 4]);
        assert_eq!(back.get("ids").unwrap().to_vec::<u8>().unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn evicts_to_disk_and_reloads() {
        let dir = temp_dir("evict");
        // Room for roughly one stage (4 KB of f32 + 3 bytes each).
        let store = TensorStore::new(&dir, 5000).unwrap();
        store.put("a", stage(1.0, 1000)).unwrap();
        store.put("b", stage(2.0, 1000)).unwrap(); // evicts "a" to disk
        assert!(dir.join("a.safetensors").exists());
        assert!(store.ram_used() <= 5000);

        store.prefetch("a");
        let a = store.get("a").unwrap(); // reload (evicts "b")
        assert_eq!(a.get("w").unwrap().to_vec::<f32>().unwrap()[0], 1.0);
        let b = store.get("b").unwrap();
        assert_eq!(b.get("w").unwrap().to_vec::<f32>().unwrap()[999], 2.0);
        assert!(store.stats().write_bytes.load(Ordering::Relaxed) > 0);
        assert!(store.stats().read_bytes.load(Ordering::Relaxed) > 0);

        store.flush().unwrap();
        let ckpt = dir.join("ckpt");
        store.copy_stages_to("", &ckpt).unwrap();
        assert!(ckpt.join("a.safetensors").exists() && ckpt.join("b.safetensors").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn concurrent_gets_share_one_load() {
        let dir = temp_dir("concurrent");
        let store = TensorStore::new(&dir, 0).unwrap(); // nothing stays in RAM
        store.put("x", stage(3.0, 10)).unwrap();
        store.flush().unwrap();
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let s = Arc::clone(&store);
                std::thread::spawn(move || s.get("x").unwrap().get("w").unwrap().to_vec::<f32>().unwrap()[0])
            })
            .collect();
        for h in handles {
            assert_eq!(h.join().unwrap(), 3.0);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn newest_version_wins_on_disk() {
        let dir = temp_dir("versions");
        let store = TensorStore::new(&dir, 0).unwrap(); // every put is evicted and written
        for v in 1..=5 {
            store.put("s", stage(v as f32, 16)).unwrap();
        }
        store.flush().unwrap();
        let back = deserialize(&std::fs::read(dir.join("s.safetensors")).unwrap()).unwrap();
        assert_eq!(back.get("w").unwrap().to_vec::<f32>().unwrap()[0], 5.0);
        assert_eq!(store.get("s").unwrap().get("w").unwrap().to_vec::<f32>().unwrap()[0], 5.0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
