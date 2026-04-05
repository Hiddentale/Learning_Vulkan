use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

const WORLDS_DIR: &str = "worlds";
const META_FILE: &str = "world.toml";

pub struct WorldMeta {
    pub name: String,
    pub seed: u32,
    pub created_at: String,
}

impl WorldMeta {
    pub fn save(&self, dir: &Path) -> anyhow::Result<()> {
        fs::create_dir_all(dir)?;
        let path = dir.join(META_FILE);
        let mut file = fs::File::create(path)?;
        writeln!(file, "name = \"{}\"", self.name)?;
        writeln!(file, "seed = {}", self.seed)?;
        writeln!(file, "created_at = \"{}\"", self.created_at)?;
        Ok(())
    }

    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        let contents = fs::read_to_string(dir.join(META_FILE))?;
        let mut name = String::new();
        let mut seed = 0u32;
        let mut created_at = String::new();
        for line in contents.lines() {
            let line = line.trim();
            if let Some(val) = line.strip_prefix("name = ") {
                name = val.trim_matches('"').to_string();
            } else if let Some(val) = line.strip_prefix("seed = ") {
                seed = val.parse()?;
            } else if let Some(val) = line.strip_prefix("created_at = ") {
                created_at = val.trim_matches('"').to_string();
            }
        }
        anyhow::ensure!(!name.is_empty(), "missing 'name' in world.toml");
        Ok(Self { name, seed, created_at })
    }
}

/// List all saved worlds sorted by name.
pub fn list_worlds() -> Vec<(PathBuf, WorldMeta)> {
    let base = Path::new(WORLDS_DIR);
    let mut worlds = Vec::new();
    let entries = match fs::read_dir(base) {
        Ok(e) => e,
        Err(_) => return worlds,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Ok(meta) = WorldMeta::load(&path) {
                worlds.push((path, meta));
            }
        }
    }
    worlds.sort_by(|a, b| a.1.name.cmp(&b.1.name));
    worlds
}

/// Create a new world directory with metadata. Returns the world directory path.
pub fn create_world(name: &str, seed: u32) -> anyhow::Result<PathBuf> {
    let dir_name = sanitize_dir_name(name);
    let dir = Path::new(WORLDS_DIR).join(&dir_name);
    anyhow::ensure!(!dir.exists(), "world directory already exists: {}", dir.display());
    let meta = WorldMeta {
        name: name.to_string(),
        seed,
        created_at: current_timestamp(),
    };
    meta.save(&dir)?;
    fs::create_dir_all(dir.join("svdag"))?;
    Ok(dir)
}

/// Returns the svdag cache path for a world directory at the given LOD level.
pub fn svdag_lod_dir(world_dir: &Path, lod_level: u32) -> PathBuf {
    if lod_level == 0 {
        world_dir.join("svdag")
    } else {
        world_dir.join(format!("svdag_lod{lod_level}"))
    }
}

fn sanitize_dir_name(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}

fn current_timestamp() -> String {
    // Simple timestamp without chrono dependency: seconds since UNIX epoch
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{secs}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    static TEST_ID: AtomicU32 = AtomicU32::new(0);

    fn temp_dir() -> PathBuf {
        let id = TEST_ID.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("manifold_meta_{}_{}", std::process::id(), id));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn save_load_roundtrip() {
        let dir = temp_dir();
        let meta = WorldMeta {
            name: "Test World".to_string(),
            seed: 123,
            created_at: "1000".to_string(),
        };
        meta.save(&dir).unwrap();
        let loaded = WorldMeta::load(&dir).unwrap();
        assert_eq!(loaded.name, "Test World");
        assert_eq!(loaded.seed, 123);
        assert_eq!(loaded.created_at, "1000");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn sanitize_removes_special_chars() {
        assert_eq!(sanitize_dir_name("My World!"), "My_World_");
        assert_eq!(sanitize_dir_name("test-world_1"), "test-world_1");
    }
}
