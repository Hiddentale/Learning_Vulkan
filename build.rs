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


fn main()  -> anyhow::Result<()> {
    let glslc_path = find_glslc()?;
    println!("Using glslc at: {:?}", glslc_path.display());
    let all_shader_source_files = read_all_entries()?;
    println!("{:?}", all_shader_source_files);
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
    for entry_result in std::fs::read_dir(shader_directory_path)? { // For each entry
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
    }
    else {
        return Ok(shader_paths);
    }
}