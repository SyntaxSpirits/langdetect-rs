use std::cell::RefCell;
use std::sync::OnceLock;

use rustc_hash::FxHashMap;
use serde_json::Value;

use crate::compat::{sum, Compat, SumMode, UnicodeVersion, UpperTable};
use crate::detector::{Detector, Language};
use crate::error::{Error, ErrorCode};
use crate::random::Seed;
use crate::text;

static BUILTIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/profiles.bin"));

const ALPHA_WIDTH: f64 = 0.05;
const ITERATION_LIMIT: usize = 1000;
const PROB_THRESHOLD: f64 = 0.1;
const CONV_THRESHOLD: f64 = 0.99999;
const BASE_FREQ: f64 = 10000.0;

/// A language profile: n-gram counts and the total number of 1-, 2- and 3-grams.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LangProfile {
    pub name: String,
    pub freq: Vec<(String, f64)>,
    pub n_words: Vec<f64>,
}

impl LangProfile {
    /// Parses the JSON format of langdetect profiles (`{"name", "freq", "n_words"}`).
    pub fn from_json(json: &str) -> Result<Self, Error> {
        let format = || Error::new(ErrorCode::FormatError, "Profile format error.");
        let v: Value = serde_json::from_str(json).map_err(|_| format())?;
        let obj = v.as_object().ok_or_else(format)?;
        if obj
            .keys()
            .any(|k| !matches!(k.as_str(), "name" | "freq" | "n_words"))
        {
            return Err(format());
        }
        let name = obj
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(format)?
            .to_string();
        let freq = match obj.get("freq") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Object(m)) => m
                .iter()
                .map(|(k, v)| v.as_f64().map(|f| (k.clone(), f)).ok_or_else(format))
                .collect::<Result<_, _>>()?,
            Some(_) => return Err(format()),
        };
        let n_words = match obj.get("n_words") {
            None | Some(Value::Null) => vec![0.0; 3],
            Some(Value::Array(a)) => a
                .iter()
                .map(|v| v.as_f64().ok_or_else(format))
                .collect::<Result<_, _>>()?,
            Some(_) => return Err(format()),
        };
        Ok(LangProfile {
            name,
            freq,
            n_words,
        })
    }
}

/// Packs an n-gram of one to three characters into a hash key.
#[inline]
fn pack(chars: &[char]) -> u64 {
    let mut k = 0u64;
    for (i, &c) in chars.iter().enumerate() {
        k |= (c as u64 + 1) << (21 * i);
    }
    k
}

/// Options of one detection, the attributes of `langdetect.detector.Detector`.
#[derive(Clone, Copy, Debug)]
pub struct DetectOptions<'a> {
    pub alpha: f64,
    pub n_trial: i64,
    pub prior_map: Option<&'a [f64]>,
}

impl Default for DetectOptions<'_> {
    fn default() -> Self {
        DetectOptions {
            alpha: 0.5,
            n_trial: 7,
            prior_map: None,
        }
    }
}

/// One n-gram's probabilities: `word_lang_prob_map[word]` of langdetect, a list of `len`
/// floats of which only `slots` are non-zero. `len` is the `langsize` given when the n-gram was
/// first added; `u32::MAX` stands for "long enough".
#[derive(Clone, Debug)]
struct Row {
    len: u32,
    slots: Vec<(u32, f64)>,
}

const UNBOUNDED: usize = u32::MAX as usize;

/// Language profiles and n-gram probabilities, the equivalent of `DetectorFactory`.
#[derive(Clone, Debug)]
pub struct DetectorFactory {
    langlist: Vec<String>,
    compat: Compat,
    index: FxHashMap<u64, u32>,
    keys: Vec<u64>,
    rows: Vec<Row>,
    /// Row + 1 of each single-character n-gram in the Basic Multilingual Plane, 0 if none.
    unigram: Vec<u32>,
    pub seed: Seed,
}

impl Default for DetectorFactory {
    fn default() -> Self {
        Self::new(Compat::default())
    }
}

impl DetectorFactory {
    /// An empty factory.
    pub fn new(compat: Compat) -> Self {
        DetectorFactory {
            langlist: Vec::new(),
            compat,
            index: FxHashMap::default(),
            keys: Vec::new(),
            rows: Vec::new(),
            unigram: Vec::new(),
            seed: Seed::Os,
        }
    }

    /// The 55 profiles shipped with langdetect 1.0.9, in file name order, with
    /// `Compat::default()`.
    pub fn builtin() -> &'static DetectorFactory {
        static FACTORY: OnceLock<DetectorFactory> = OnceLock::new();
        FACTORY.get_or_init(|| {
            let mut f = DetectorFactory::default();
            f.load_builtin(&Self::builtin_files()).unwrap();
            f
        })
    }

    /// File names of the built-in profiles (`af`, `ar`, ..., `zh-tw`).
    pub fn builtin_files() -> Vec<String> {
        BuiltinProfiles::new().map(|p| p.file.to_string()).collect()
    }

    /// `DetectorFactory.load_profile` on the built-in profiles: adds them in the given order
    /// (langdetect uses `os.listdir` order) at indices 0, 1, ... with `langsize` equal to the
    /// number of files. On failure, returns how many profiles were added and the error.
    pub fn load_builtin<S: AsRef<str>>(&mut self, files: &[S]) -> Result<(), (usize, Error)> {
        let all: Vec<BuiltinProfile> = BuiltinProfiles::new().collect();
        for (index, file) in files.iter().enumerate() {
            let file = file.as_ref();
            let profile = all.iter().find(|p| p.file == file).ok_or_else(|| {
                let message = format!("Not found profile: {file}");
                (index, Error::new(ErrorCode::NeedLoadProfileError, message))
            })?;
            self.add_profile_at(&profile.to_profile(), index, files.len())
                .map_err(|e| (index, e))?;
        }
        Ok(())
    }

    /// Adds a profile as the next language.
    pub fn add_profile(&mut self, profile: &LangProfile) -> Result<(), Error> {
        self.add_profile_at(profile, self.langlist.len(), UNBOUNDED)
    }

    /// Parses and adds one profile in langdetect's JSON format as the next language.
    pub fn add_json_profile(&mut self, json: &str) -> Result<(), Error> {
        self.add_profile(&LangProfile::from_json(json)?)
    }

    /// `DetectorFactory.add_profile(profile, index, langsize)`, step by step: the language is
    /// appended, then each n-gram's probability is stored at `index` of its row, a new row
    /// having `langsize` slots. As in langdetect, an error leaves the steps before it applied.
    pub fn add_profile_at(
        &mut self,
        profile: &LangProfile,
        index: usize,
        langsize: usize,
    ) -> Result<(), Error> {
        if self.langlist.contains(&profile.name) {
            return Err(Error::new(
                ErrorCode::DuplicateLangError,
                "Duplicate the same language profile.",
            ));
        }
        self.langlist.push(profile.name.clone());
        let langsize = langsize.min(UNBOUNDED) as u32;
        let mut chars = Vec::with_capacity(3);
        for (word, count) in &profile.freq {
            chars.clear();
            chars.extend(word.chars().take(4));
            if !(1..=3).contains(&chars.len()) {
                continue;
            }
            let row = self.row_for(&chars, langsize);
            let n = *profile
                .n_words
                .get(chars.len() - 1)
                .ok_or(Error::index("list index out of range"))?;
            if n == 0.0 {
                return Err(Error::zero_division());
            }
            let prob = count / n;
            let row = &mut self.rows[row as usize];
            if index >= row.len as usize {
                return Err(Error::index("list assignment index out of range"));
            }
            let slot = index as u32;
            match row.slots.iter_mut().find(|(s, _)| *s == slot) {
                Some(entry) => entry.1 = prob,
                None => row.slots.push((slot, prob)),
            }
        }
        Ok(())
    }

    /// Creates the row of a one- to three-character n-gram if it does not exist, as langdetect
    /// does for each n-gram of a profile before computing its probability.
    #[doc(hidden)]
    pub fn touch_ngram(&mut self, word: &str, langsize: usize) {
        let chars: Vec<char> = word.chars().take(4).collect();
        if (1..=3).contains(&chars.len()) {
            self.row_for(&chars, langsize.min(UNBOUNDED) as u32);
        }
    }

    fn row_for(&mut self, chars: &[char], len: u32) -> u32 {
        let key = pack(chars);
        if let Some(&row) = self.index.get(&key) {
            return row;
        }
        let row = self.rows.len() as u32;
        self.index.insert(key, row);
        self.keys.push(key);
        self.rows.push(Row {
            len,
            slots: Vec::new(),
        });
        if chars.len() == 1 && (chars[0] as u32) < 0x10000 {
            if self.unigram.is_empty() {
                self.unigram = vec![0; 0x10000];
            }
            self.unigram[chars[0] as usize] = row + 1;
        }
        row
    }

    pub fn clear(&mut self) {
        let seed = std::mem::take(&mut self.seed);
        *self = DetectorFactory::new(self.compat);
        self.seed = seed;
    }

    pub fn langlist(&self) -> &[String] {
        &self.langlist
    }

    pub fn compat(&self) -> Compat {
        self.compat
    }

    /// `DetectorFactory.create()`.
    pub fn create(&self) -> Result<Detector<'_>, Error> {
        if self.langlist.is_empty() {
            return Err(Error::new(
                ErrorCode::NeedLoadProfileError,
                "Need to load profiles.",
            ));
        }
        Ok(Detector::new(self))
    }

    /// `langdetect.detect(text)` with this factory.
    pub fn detect(&self, text: &str) -> Result<String, Error> {
        let mut d = self.create()?;
        d.append(text);
        d.detect()
    }

    /// `langdetect.detect_langs(text)` with this factory.
    pub fn detect_langs(&self, text: &str) -> Result<Vec<Language>, Error> {
        let mut d = self.create()?;
        d.append(text);
        d.get_probabilities()
    }

    /// `NGram`-based feature extraction of `Detector._extract_ngrams`: row ids of the
    /// n-grams of `text` that occur in some profile.
    pub(crate) fn extract(&self, text: &str) -> Vec<u32> {
        let upper = UpperTable::get(self.compat.unicode);
        let mut result = Vec::with_capacity(text.len() * 2);
        let mut grams = [' '; 3];
        let mut caps = [false; 3];
        let mut len = 1usize;
        let mut capitalword = false;
        for raw in text.chars() {
            let ch = text::normalize(raw);
            let last_upper = caps[len - 1];
            if grams[len - 1] == ' ' {
                grams[0] = ' ';
                caps[0] = false;
                len = 1;
                capitalword = false;
                if ch == ' ' {
                    continue;
                }
            } else if len >= 3 {
                grams[0] = grams[1];
                grams[1] = grams[2];
                caps[0] = caps[1];
                caps[1] = caps[2];
                len = 2;
            }
            let is_upper = upper.is_upper(ch);
            grams[len] = ch;
            caps[len] = is_upper;
            len += 1;
            if is_upper {
                if last_upper {
                    capitalword = true;
                }
            } else {
                capitalword = false;
            }
            if capitalword {
                continue;
            }
            for n in 1..=3 {
                if len < n {
                    break;
                }
                let w = &grams[len - n..len];
                if n == 1 {
                    if w[0] == ' ' {
                        continue;
                    }
                    let cp = w[0] as usize;
                    if cp < 0x10000 {
                        match self.unigram.get(cp) {
                            Some(&r) if r != 0 => result.push(r - 1),
                            _ => {}
                        }
                        continue;
                    }
                }
                if let Some(&row) = self.index.get(&pack(w)) {
                    result.push(row);
                }
            }
        }
        result
    }

    /// `Detector._detect_block` after cleaning: the averaged probability of every language.
    ///
    /// On `Error::ZeroDivision` and `Error::Index` raised during the walk, the error carries
    /// the probabilities accumulated so far, which langdetect keeps in `Detector.langprob`.
    pub fn detect_block(
        &self,
        text: &str,
        options: &DetectOptions,
        seed: &Seed,
    ) -> Result<Vec<f64>, Error> {
        let ngrams = self.extract(text);
        if ngrams.is_empty() {
            return Err(Error::new(
                ErrorCode::CantDetectError,
                "No features in text.",
            ));
        }
        // Each step is the vectorisable `prob[i] *= weight + row[i]` of langdetect, on a dense
        // row built the first time its n-gram is drawn.
        thread_local! {
            static DENSE: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
        }
        DENSE.with_borrow_mut(|dense| {
            dense.clear();
            self.random_walk(dense, &ngrams, options, seed)
        })
    }

    fn random_walk(
        &self,
        dense: &mut Vec<f64>,
        ngrams: &[u32],
        options: &DetectOptions,
        seed: &Seed,
    ) -> Result<Vec<f64>, Error> {
        const UNBUILT: u32 = u32::MAX;
        let nl = self.langlist.len();
        // The length of `prob`: the prior's, or one per language.
        let width = options.prior_map.map_or(nl, <[f64]>::len);
        let mut built: FxHashMap<u32, u32> = FxHashMap::default();
        let mut slots = vec![UNBUILT; ngrams.len()];
        let mut langprob = vec![0.0f64; nl];
        let mut rng = seed.rng();
        let mut prob = vec![0.0f64; width];
        let n_trial = options.n_trial.max(0);
        for _ in 0..n_trial {
            match options.prior_map {
                Some(prior) => prob.copy_from_slice(prior),
                None if nl == 0 => return Err(Error::zero_division_with(langprob)),
                None => prob.fill(1.0 / nl as f64),
            }
            let alpha = options.alpha + rng.gauss() * ALPHA_WIDTH;
            let weight = alpha / BASE_FREQ;
            let mut i = 0usize;
            loop {
                let pick = rng.randbelow(ngrams.len());
                let mut slot = slots[pick];
                if slot == UNBUILT {
                    let row = &self.rows[ngrams[pick] as usize];
                    if (row.len as usize) < width {
                        return Err(Error::index_with("list index out of range", langprob));
                    }
                    slot = *built.entry(ngrams[pick]).or_insert_with(|| {
                        let start = dense.len();
                        dense.resize(start + width, 0.0);
                        for &(s, p) in &row.slots {
                            if (s as usize) < width {
                                dense[start + s as usize] = p;
                            }
                        }
                        (start / width.max(1)) as u32
                    });
                    slots[pick] = slot;
                }
                let row = &dense[slot as usize * width..(slot as usize + 1) * width];
                for (p, &r) in prob.iter_mut().zip(row) {
                    *p *= weight + r;
                }
                if i % 5 == 0 {
                    let total = sum(&prob, self.compat.sum);
                    if total == 0.0 && !prob.is_empty() {
                        return Err(Error::zero_division_with(langprob));
                    }
                    for p in prob.iter_mut() {
                        *p /= total;
                    }
                    let maxp = max_prob(&prob);
                    if maxp > CONV_THRESHOLD || i >= ITERATION_LIMIT {
                        break;
                    }
                }
                i += 1;
            }
            for (j, lp) in langprob.iter_mut().enumerate() {
                let Some(&p) = prob.get(j) else {
                    return Err(Error::index_with("list index out of range", langprob));
                };
                *lp += p / n_trial as f64;
            }
        }
        Ok(langprob)
    }

    /// `Detector._sort_probability`: languages above 0.1, most probable first.
    pub fn sort_probability(&self, langprob: &[f64]) -> Vec<Language> {
        let mut v: Vec<Language> = self
            .langlist
            .iter()
            .zip(langprob)
            .filter(|(_, &p)| p > PROB_THRESHOLD)
            .map(|(lang, &prob)| Language {
                lang: lang.clone(),
                prob,
            })
            .collect();
        v.sort_by(|a, b| {
            b.prob
                .partial_cmp(&a.prob)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        v
    }

    /// Serialises the profiles, the n-gram table and the compatibility settings (not the seed).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = b"LDF1".to_vec();
        out.push(self.compat.sum as u8);
        out.push(self.compat.unicode as u8);
        out.extend_from_slice(&(self.langlist.len() as u32).to_le_bytes());
        for lang in &self.langlist {
            out.extend_from_slice(&(lang.len() as u32).to_le_bytes());
            out.extend_from_slice(lang.as_bytes());
        }
        out.extend_from_slice(&(self.rows.len() as u32).to_le_bytes());
        for (key, row) in self.keys.iter().zip(&self.rows) {
            out.extend_from_slice(&key.to_le_bytes());
            out.extend_from_slice(&row.len.to_le_bytes());
            out.extend_from_slice(&(row.slots.len() as u32).to_le_bytes());
            for &(s, p) in &row.slots {
                out.extend_from_slice(&s.to_le_bytes());
                out.extend_from_slice(&p.to_bits().to_le_bytes());
            }
        }
        out
    }

    /// Restores a factory saved by [`DetectorFactory::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let bad = || Error::new(ErrorCode::FormatError, "Invalid serialised factory.");
        let mut r = Reader { buf: bytes, pos: 0 };
        if r.take(4).ok_or_else(bad)? != b"LDF1" {
            return Err(bad());
        }
        let sum = match r.take(1).ok_or_else(bad)?[0] {
            0 => SumMode::Sequential,
            1 => SumMode::Neumaier,
            _ => return Err(bad()),
        };
        let unicode = UnicodeVersion::from_index(r.take(1).ok_or_else(bad)?[0]).ok_or_else(bad)?;
        let mut f = DetectorFactory::new(Compat { sum, unicode });
        for _ in 0..r.u32().ok_or_else(bad)? {
            let len = r.u32().ok_or_else(bad)? as usize;
            let name = std::str::from_utf8(r.take(len).ok_or_else(bad)?).map_err(|_| bad())?;
            f.langlist.push(name.to_string());
        }
        for _ in 0..r.u32().ok_or_else(bad)? {
            let key = r.u64().ok_or_else(bad)?;
            let chars = unpack(key).ok_or_else(bad)?;
            let len = r.u32().ok_or_else(bad)?;
            let row = f.row_for(&chars, len);
            if row as usize != f.rows.len() - 1 {
                return Err(bad());
            }
            for _ in 0..r.u32().ok_or_else(bad)? {
                let s = r.u32().ok_or_else(bad)?;
                let p = f64::from_bits(r.u64().ok_or_else(bad)?);
                f.rows[row as usize].slots.push((s, p));
            }
        }
        if r.pos != bytes.len() {
            return Err(bad());
        }
        Ok(f)
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.buf.get(self.pos..self.pos.checked_add(n)?)?;
        self.pos += n;
        Some(s)
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
}

fn unpack(key: u64) -> Option<Vec<char>> {
    let mut chars = Vec::with_capacity(3);
    for i in 0..3 {
        let v = (key >> (21 * i)) & 0x1f_ffff;
        if v == 0 {
            break;
        }
        chars.push(char::from_u32(v as u32 - 1)?);
    }
    if chars.is_empty() || pack(&chars) != key {
        return None;
    }
    Some(chars)
}

/// `maxp` of `Detector._normalize_prob` (`if maxp < p: maxp = p`, starting at 0.0). It is
/// only compared with the convergence threshold, so the order of comparisons and the sign of
/// zero do not matter, and NaN never wins in either version.
fn max_prob(prob: &[f64]) -> f64 {
    let mut lanes = [0.0f64; 4];
    let chunks = prob.chunks_exact(4);
    let rest = chunks.remainder();
    for c in chunks {
        for (m, &p) in lanes.iter_mut().zip(c) {
            *m = m.max(p);
        }
    }
    let mut maxp = lanes[0].max(lanes[1]).max(lanes[2].max(lanes[3]));
    for &p in rest {
        maxp = maxp.max(p);
    }
    maxp
}

struct BuiltinProfile {
    file: &'static str,
    name: &'static str,
    n_words: [u64; 3],
    entries: &'static [u8],
    count: u32,
}

impl BuiltinProfile {
    fn to_profile(&self) -> LangProfile {
        let mut freq = Vec::with_capacity(self.count as usize);
        let mut pos = 0;
        for _ in 0..self.count {
            let len = self.entries[pos] as usize;
            let key = std::str::from_utf8(&self.entries[pos + 1..pos + 1 + len]).unwrap();
            pos += 1 + len;
            let count = read_varint(self.entries, &mut pos);
            freq.push((key.to_string(), count as f64));
        }
        LangProfile {
            name: self.name.to_string(),
            freq,
            n_words: self.n_words.iter().map(|&n| n as f64).collect(),
        }
    }
}

fn read_varint(buf: &[u8], pos: &mut usize) -> u64 {
    let mut v = 0u64;
    let mut shift = 0;
    loop {
        let b = buf[*pos];
        *pos += 1;
        v |= ((b & 0x7f) as u64) << shift;
        if b < 0x80 {
            return v;
        }
        shift += 7;
    }
}

struct BuiltinProfiles {
    pos: usize,
    left: u16,
}

impl BuiltinProfiles {
    fn new() -> Self {
        BuiltinProfiles {
            pos: 2,
            left: u16::from_le_bytes([BUILTIN[0], BUILTIN[1]]),
        }
    }

    fn str(&mut self) -> &'static str {
        let len = BUILTIN[self.pos] as usize;
        let s = std::str::from_utf8(&BUILTIN[self.pos + 1..self.pos + 1 + len]).unwrap();
        self.pos += 1 + len;
        s
    }
}

impl Iterator for BuiltinProfiles {
    type Item = BuiltinProfile;

    fn next(&mut self) -> Option<BuiltinProfile> {
        if self.left == 0 {
            return None;
        }
        self.left -= 1;
        let file = self.str();
        let name = self.str();
        let mut n_words = [0u64; 3];
        for n in &mut n_words {
            *n = read_varint(BUILTIN, &mut self.pos);
        }
        let count = u32::from_le_bytes(BUILTIN[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        let start = self.pos;
        for _ in 0..count {
            let len = BUILTIN[self.pos] as usize;
            self.pos += 1 + len;
            read_varint(BUILTIN, &mut self.pos);
        }
        Some(BuiltinProfile {
            file,
            name,
            n_words,
            entries: &BUILTIN[start..self.pos],
            count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(name: &str, freq: &[(&str, f64)]) -> LangProfile {
        LangProfile {
            name: name.into(),
            freq: freq.iter().map(|&(w, c)| (w.into(), c)).collect(),
            n_words: vec![10.0, 10.0, 10.0],
        }
    }

    #[test]
    fn empty_factory_has_no_features() {
        let f = DetectorFactory::default();
        let err = f
            .detect_block("abc", &DetectOptions::default(), &Seed::Int(0))
            .unwrap_err();
        assert_eq!(err.code(), Some(ErrorCode::CantDetectError));
        assert!(f.create().is_err());
    }

    #[test]
    fn prior_of_any_length_is_an_error_not_a_panic() {
        let f = DetectorFactory::builtin();
        for prior in [&[][..], &[1.0; 54][..], &[1.0; 56][..]] {
            let options = DetectOptions {
                prior_map: Some(prior),
                ..Default::default()
            };
            let err = f
                .detect_block("hello", &options, &Seed::Int(0))
                .unwrap_err();
            assert!(matches!(err, Error::Index { .. }), "{err:?}");
            assert_eq!(err.langprob().map(<[f64]>::len), Some(55));
        }
    }

    #[test]
    fn profile_index_selects_the_slot() {
        let mut f = DetectorFactory {
            seed: Seed::Int(0),
            ..Default::default()
        };
        f.add_profile_at(&profile("x", &[("a", 9.0)]), 1, 2)
            .unwrap();
        f.add_profile_at(&profile("y", &[("a", 1.0)]), 0, 2)
            .unwrap();
        assert_eq!(f.detect("a").unwrap(), "y");
        let err = f
            .add_profile_at(&profile("z", &[("b", 1.0)]), 2, 2)
            .unwrap_err();
        assert!(matches!(err, Error::Index { .. }));
        assert_eq!(f.langlist(), ["x", "y", "z"]);
    }

    #[test]
    fn serialisation_round_trips() {
        let f = DetectorFactory::builtin();
        let g = DetectorFactory::from_bytes(&f.to_bytes()).unwrap();
        assert_eq!(g.langlist(), f.langlist());
        let options = DetectOptions::default();
        let text = "Ceci est une phrase en français.";
        assert_eq!(
            g.detect_block(text, &options, &Seed::Int(1)),
            f.detect_block(text, &options, &Seed::Int(1))
        );
        assert!(DetectorFactory::from_bytes(b"LDF1").is_err());
    }
}
