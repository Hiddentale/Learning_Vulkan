/*
1. Check if glslc is available
   - If not, panic with helpful message

2. Define shader directory path ("src/shaders")

3. Read all entries in shader directory
   - For each entry:
     - Check if it's a file
     - Check if extension is "vert" or "frag"
     - If yes: this is a shader source file

4. For each shader source file:
   - Tell Cargo to watch this file
   - Build the output path
   - Check if output exists and if source is newer
   - If we need to compile:
     - Run glslc with input and output paths
     - Check if compilation succeeded
     - If failed, panic with error message
     */

fn main() -> anyhow::Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    let glslc_path = find_glslc()?;
    let all_shader_source_files = read_all_entries()?;
    let a = process_shaders(all_shader_source_files);

    Ok(())
}

fn find_glslc() -> anyhow::Result<std::path::PathBuf> {
    let environment_variable_path = std::env::var("PATH")?;

    for directory in std::env::split_paths(&environment_variable_path) {
        if let Ok(files_in_directory) = std::fs::read_dir(&directory) {
            for file in files_in_directory {
                if let Ok(entry) = file {
                    if entry.file_name() == "glslc.exe" {
                        return Ok(entry.path());
                    }
                }
            }
        }
    }
    anyhow::bail!("glslc.exe not found!")
}

fn read_all_entries() -> anyhow::Result<Vec<std::path::PathBuf>> {
    let mut shader_paths = Vec::new();
    let shader_directory_path = "src/shaders";
    for entry_result in std::fs::read_dir(shader_directory_path)? {
        if let Ok(entry) = entry_result {
            if entry.path().is_file() {
                if entry.path().extension().unwrap() == "vert" || entry.path().extension().unwrap() == "frag" {
                    shader_paths.push(entry.path())
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

fn process_shaders(shader_paths: Vec<std::path::PathBuf>) -> anyhow::Result<()> {
    for shader_path in shader_paths {
        println!("cargo:rerun-if-changed={:?}", shader_path.display());
        let shader_modified_date = std::fs::metadata(&shader_path)?.modified();
        println!("shader_modified_date: {:?}", shader_modified_date);
        let extension_type = shader_path.extension();
        let compiled_shader_path = format!("src/shaders/{}.spv", extension_type.unwrap().to_str().unwrap());
        println!("compiled_shader_path: {:?}", compiled_shader_path);
        let compiled_shader_modified_date = std::fs::metadata(&compiled_shader_path)?.modified();
        println!("compiled_shader_modified_date: {:?}", compiled_shader_modified_date);
        if shader_modified_date? > compiled_shader_modified_date? {
            println!("Files out of sync! Time to compile!");
            let output = std::process::Command::new("glslc")
                .arg(shader_path)
                .arg("-o")
                .arg(compiled_shader_path)
                .output()?;
            if !output.status.success() {
                anyhow::bail!("glslc runtime error!");
            }
        }
    }
    Ok(())
}
