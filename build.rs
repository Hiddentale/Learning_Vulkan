use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    println!("cargo:rerun-if-changed=build.rs");

    if !is_glsl_compiler_available() {
        println!("cargo:warning=glslc not found — skipping shader compilation. Using pre-compiled .spv files.");
        return Ok(());
    }

    let all_shader_source_files = discover_shader_files()?;
    process_shaders(all_shader_source_files)?;

    Ok(())
}

fn is_glsl_compiler_available() -> bool {
    std::process::Command::new("glslc").arg("--version").output().is_ok()
}

fn discover_shader_files() -> anyhow::Result<Vec<PathBuf>> {
    let mut shader_paths = Vec::new();
    let shader_directory_path = "src/shaders";

    for entry_result in std::fs::read_dir(shader_directory_path)? {
        let entry = entry_result?.path();
        if entry.is_file() {
            if let Some(extension) = entry.extension() {
                if matches!(extension.to_str(), Some("vert" | "frag" | "comp" | "task" | "mesh")) {
                    shader_paths.push(entry)
                }
            }
        }
    }
    if shader_paths.is_empty() {
        anyhow::bail!("No shaders found!")
    } else {
        Ok(shader_paths)
    }
}

fn process_shaders(shader_paths: Vec<PathBuf>) -> anyhow::Result<()> {
    for shader_path in shader_paths {
        println!("cargo:rerun-if-changed={}", shader_path.display());

        let shader_modified_date = std::fs::metadata(&shader_path)?.modified();
        let compiled_shader_path = format!("{}.spv", shader_path.display());

        let needs_recompile = match std::fs::metadata(&compiled_shader_path) {
            Ok(metadata) => {
                let compiled_time = metadata.modified()?;
                shader_modified_date? > compiled_time
            }
            Err(_) => true,
        };
        if needs_recompile {
            let mut cmd = std::process::Command::new("glslc");
            // Mesh/task shaders require Vulkan 1.2+ target environment
            if let Some(ext) = shader_path.extension() {
                if ext == "task" || ext == "mesh" {
                    cmd.arg("--target-env=vulkan1.2");
                }
            }
            let output = cmd
                .arg(&shader_path)
                .arg("-o")
                .arg(compiled_shader_path)
                .output()?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                anyhow::bail!("Shader compilation failed: \n{}", stderr);
            }
        }
    }
    Ok(())
}
