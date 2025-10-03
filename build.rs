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


fn main() {
    println!("Hello from build script!");
}