//! # Non-blocking `find`-style picker
//!
//! Iterate over directories to populate the picker, but do not block so that matching can be done
//! while the picker is populated.
use std::{env::args, path::PathBuf, thread::spawn};

use anyhow::Result;
use nucleo_picker::{nucleo::Config, Picker};
use walkdir::WalkDir;

fn main() -> Result<()> {
    // See the nucleo configuration for more options:
    //   https://docs.rs/nucleo/latest/nucleo/struct.Config.html
    let config = Config::DEFAULT.match_paths();

    // Initialize a picker with the provided configuration
    let mut picker =
        Picker::with_config(config).with_previewer("/usr/bin/bat %t --color=never".to_string());

    // "argument parsing"
    let root: PathBuf = match args().nth(1) {
        Some(path) => path.into(),
        None => ".".into(),
    };

    // populate from a separate thread to avoid locking the picker interface
    let injector = picker.injector();
    spawn(move || {
        for entry in WalkDir::new(root).into_iter().filter_map(Result::ok) {
            let _ = injector.push(entry, |e, cols| {
                // the picker only has one column; fill it with the match text
                cols[0] = e.path().display().to_string().into();
            });
        }
    });

    match picker.pick()? {
        Some(entry) => {
            // the matched `entry` is &DirEntry
            println!(
                "Name of selected file: '{}'",
                entry.file_name().to_string_lossy()
            );
        }
        None => {
            println!("No file selected!");
        }
    }

    Ok(())
}
