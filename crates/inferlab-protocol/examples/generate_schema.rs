use inferlab_protocol::{measurement_schema, protocol_schema};
use std::error::Error;
use std::io::{Error as IoError, ErrorKind};
use std::path::PathBuf;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let adapter_output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "missing adapter output path"))?;
    let measurement_output = std::env::args_os()
        .nth(2)
        .map(PathBuf::from)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "missing measurement output path"))?;
    write_schema(adapter_output, &protocol_schema())?;
    write_schema(measurement_output, &measurement_schema())?;
    Ok(())
}

fn write_schema(output: PathBuf, schema: &schemars::Schema) -> Result<(), Box<dyn Error>> {
    let mut rendered = serde_json::to_string_pretty(schema)?;
    rendered.push('\n');
    std::fs::write(output, rendered)?;
    Ok(())
}
