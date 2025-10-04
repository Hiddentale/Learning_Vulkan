//! Build script for automatic shader compilation.
//!
//! This script runs before the main crate is compiled and handles:
//! - Detecting the GLSL compiler (glslc from Vulkan SDK)
//! - Discovering shader source files in `src/shaders/`
//! - Compiling GLSL to SPIR-V bytecode
//! - Incremental compilation (only recompile when source changes)
//! - Integration with Cargo's rebuild system

use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    // Rebuild if this build script changes
    println!("cargo:rerun-if-changed=build.rs");

    validate_glsl_compiler()?;
    let all_shader_source_files = discover_shader_files()?;
    process_shaders(all_shader_source_files)?;

    Ok(())
}

/// Validates that the GLSL compiler (glslc) is available.
///
/// This checks if `glslc` is in the system PATH by attempting to run it
/// with the `--version` flag. If the compiler is not found, the build fails
/// with a helpful error message.
///
/// # Errors
///
/// Returns an error if glslc cannot be found or executed.
fn validate_glsl_compiler() -> anyhow::Result<()> {
    let compiler_exists = std::process::Command::new("glslc").arg("--version").output().is_ok();
    if !compiler_exists {
        anyhow::bail!(
            "glslc not found in PATH. Please install the Vulkan SDK.\n\
             Download from: https://vulkan.lunarg.com/"
        );
    }
    Ok(())
}

/// Discovers all shader source files in the `src/shaders/` directory.
///
/// Scans for files with `.vert` (vertex) and `.frag` (fragment) extensions.
/// Only files (not directories) are included in the result.
///
/// # Returns
///
/// A vector of paths to shader source files.
///
/// # Errors
///
/// Returns an error if:
/// - The shader directory cannot be read
/// - No shader files are found in the directory
fn discover_shader_files() -> anyhow::Result<Vec<PathBuf>> {
    let mut shader_paths = Vec::new();
    let shader_directory_path = "src/shaders";

    for entry_result in std::fs::read_dir(shader_directory_path)? {
        let entry = entry_result?.path();
        if entry.is_file() {
            if let Some(extension) = entry.extension() {
                if extension == "vert" || extension == "frag" {
                    shader_paths.push(entry)
                }
            }
        }
    }
    if shader_paths.is_empty() {
        anyhow::bail!("No shaders found!")
    } else {
        return Ok(shader_paths);
    }
}

/// Compiles shader source files to SPIR-V bytecode.
///
/// For each shader source file:
/// 1. Registers it with Cargo's rebuild system (recompile if it changes)
/// 2. Checks if recompilation is needed (source is newer than output)
/// 3. Invokes glslc to compile GLSL → SPIR-V
/// 4. Verifies compilation succeeded
///
/// Output files are named by appending `.spv` to the source filename.
/// For example: `shader.vert` → `shader.vert.spv`
///
/// # Arguments
///
/// * `shader_paths` - Paths to shader source files to compile
///
/// # Errors
///
/// Returns an error if:
/// - File metadata cannot be read
/// - Shader compilation fails (syntax errors, etc.)
/// - glslc cannot be executed
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
            // Compile GLSL to SPIR-V
            let output = std::process::Command::new("glslc")
                .arg(shader_path)
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
