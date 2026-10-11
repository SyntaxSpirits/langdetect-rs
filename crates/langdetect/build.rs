use std::env;
use std::fs;
use std::path::PathBuf;

fn varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn main() {
    let dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("profiles");
    println!("cargo:rerun-if-changed={}", dir.display());
    let mut files: Vec<String> = fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|f| !f.starts_with('.'))
        .collect();
    files.sort();

    let mut out = Vec::new();
    out.extend_from_slice(&(files.len() as u16).to_le_bytes());
    for file in &files {
        let text = fs::read_to_string(dir.join(file)).unwrap();
        let p: serde_json::Value = serde_json::from_str(&text).unwrap();
        let name = p["name"].as_str().unwrap();
        out.push(file.len() as u8);
        out.extend_from_slice(file.as_bytes());
        out.push(name.len() as u8);
        out.extend_from_slice(name.as_bytes());
        let n_words = p["n_words"].as_array().unwrap();
        assert_eq!(n_words.len(), 3, "{file}");
        for n in n_words {
            varint(&mut out, n.as_u64().unwrap());
        }
        let freq = p["freq"].as_object().unwrap();
        let entries: Vec<(&String, u64)> = freq
            .iter()
            .filter(|(k, _)| (1..=3).contains(&k.chars().count()))
            .map(|(k, v)| (k, v.as_u64().unwrap()))
            .collect();
        out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for (key, count) in entries {
            out.push(key.len() as u8);
            out.extend_from_slice(key.as_bytes());
            varint(&mut out, count);
        }
    }
    let dest = PathBuf::from(env::var("OUT_DIR").unwrap()).join("profiles.bin");
    fs::write(dest, out).unwrap();
}
