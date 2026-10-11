//! Text preprocessing of `Detector.append`, `Detector.cleaning_text` and `NGram`.

use std::borrow::Cow;
use std::sync::OnceLock;

use regex::Regex;

use crate::tables::{BLOCKS, CJK_MAP, LATIN1_EXCLUDED, VI_FROM, VI_MARKS, VI_TO};

const BASIC_LATIN: u8 = 1;
const LATIN_1_SUPPLEMENT: u8 = 2;
const LATIN_EXTENDED_B: u8 = 4;
const ARABIC: u8 = 13;
const LATIN_EXTENDED_ADDITIONAL: u8 = 66;
const GENERAL_PUNCTUATION: u8 = 68;
const HIRAGANA: u8 = 104;
const KATAKANA: u8 = 105;
const BOPOMOFO: u8 = 106;
const BOPOMOFO_EXTENDED: u8 = 109;
const CJK_UNIFIED_IDEOGRAPHS: u8 = 116;
const HANGUL_SYLLABLES: u8 = 140;

fn unicode_block(cp: u32) -> Option<u8> {
    if cp <= 0x7F {
        return Some(BASIC_LATIN);
    }
    let (mut be, mut en) = (0isize, BLOCKS.len() as isize - 1);
    while be <= en {
        let mid = ((be + en) >> 1) as usize;
        let (name, start, end) = BLOCKS[mid];
        if start <= cp && cp <= end {
            return Some(name);
        }
        if cp < start {
            en = mid as isize - 1;
        } else {
            be = mid as isize + 1;
        }
    }
    None
}

/// `NGram.normalize`.
fn normalize_char(c: char) -> char {
    match unicode_block(c as u32) {
        Some(BASIC_LATIN) => {
            if c < 'A' || ('Z' < c && c < 'a') || 'z' < c {
                ' '
            } else {
                c
            }
        }
        Some(LATIN_1_SUPPLEMENT) => {
            if LATIN1_EXCLUDED.contains(&c) {
                ' '
            } else {
                c
            }
        }
        Some(LATIN_EXTENDED_B) => match c {
            '\u{219}' => '\u{15f}',
            '\u{21b}' => '\u{163}',
            _ => c,
        },
        Some(GENERAL_PUNCTUATION) => ' ',
        Some(ARABIC) => {
            if c == '\u{6cc}' {
                '\u{64a}'
            } else {
                c
            }
        }
        Some(LATIN_EXTENDED_ADDITIONAL) => {
            if c >= '\u{1ea0}' {
                '\u{1ec3}'
            } else {
                c
            }
        }
        Some(HIRAGANA) => '\u{3042}',
        Some(KATAKANA) => '\u{30a2}',
        Some(BOPOMOFO | BOPOMOFO_EXTENDED) => '\u{3105}',
        Some(CJK_UNIFIED_IDEOGRAPHS) => {
            match CJK_MAP.binary_search_by_key(&(c as u32), |&(k, _)| k) {
                Ok(i) => char::from_u32(CJK_MAP[i].1).unwrap(),
                Err(_) => c,
            }
        }
        Some(HANGUL_SYLLABLES) => '\u{ac00}',
        _ => c,
    }
}

fn bmp_table() -> &'static [char] {
    static TABLE: OnceLock<Box<[char]>> = OnceLock::new();
    TABLE.get_or_init(|| {
        (0u32..0x10000)
            .map(|cp| char::from_u32(cp).map_or(' ', normalize_char))
            .collect()
    })
}

/// `NGram.normalize`.
#[inline]
pub fn normalize(c: char) -> char {
    let cp = c as u32;
    if cp < 0x10000 {
        bmp_table()[cp as usize]
    } else {
        normalize_char(c)
    }
}

/// `NGram.normalize_vi`: Vietnamese letter + combining mark into the precomposed letter.
fn normalize_vi(text: &str) -> Cow<'_, str> {
    if !text.chars().any(|c| VI_MARKS.contains(&c)) {
        return Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut it = text.chars().peekable();
    while let Some(c) = it.next() {
        if let Some(a) = VI_FROM.iter().position(|&x| x == c) {
            if let Some(&next) = it.peek() {
                if let Some(d) = VI_MARKS.iter().position(|&x| x == next) {
                    out.push(VI_TO[d][a]);
                    it.next();
                    continue;
                }
            }
        }
        out.push(c);
    }
    Cow::Owned(out)
}

/// The tables of `NGram.normalize_vi`: `TO_NORMALIZE_VI_CHARS`, `DMARK_CLASS` and
/// `NORMALIZED_VI_CHARS`.
pub fn vietnamese_tables() -> (String, String, Vec<String>) {
    (
        VI_FROM.iter().collect(),
        VI_MARKS.iter().collect(),
        VI_TO.iter().map(|row| row.iter().collect()).collect(),
    )
}

fn url_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"https?://[-_.?&~;+=/#0-9A-Za-z]{1,2076}").unwrap())
}

fn mail_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"[-_.0-9A-Za-z]{1,64}@[-_0-9A-Za-z]{1,255}[-_.0-9A-Za-z]{1,255}").unwrap()
    })
}

/// The text one `Detector.append(text)` call adds: URLs and e-mail addresses replaced by a
/// space, Vietnamese normalised, cut to `max_text_length` characters, runs of spaces collapsed.
pub fn prepare(text: &str, max_text_length: i64) -> String {
    let mut t = Cow::Borrowed(text);
    if t.contains("http") {
        if let Cow::Owned(s) = url_re().replace_all(&t, " ") {
            t = Cow::Owned(s);
        }
    }
    if t.contains('@') {
        if let Cow::Owned(s) = mail_re().replace_all(&t, " ") {
            t = Cow::Owned(s);
        }
    }
    let t = match normalize_vi(&t) {
        Cow::Owned(s) => Cow::Owned(s),
        Cow::Borrowed(_) => t,
    };
    truncate(&t, max_text_length)
}

/// The last step of `Detector.append`: the first `max_text_length` characters, with each run
/// of spaces collapsed to one.
pub fn truncate(text: &str, max_text_length: i64) -> String {
    let limit = usize::try_from(max_text_length).unwrap_or(0);
    let mut out = String::with_capacity(text.len().min(limit.saturating_mul(4)));
    let mut pre: Option<char> = None;
    for ch in text.chars().take(limit) {
        if ch != ' ' || pre != Some(' ') {
            out.push(ch);
        }
        pre = Some(ch);
    }
    out
}

/// `Detector.cleaning_text`: drops Latin letters when the text is mostly non-Latin.
/// Returns `None` when the text stays unchanged.
pub fn clean(text: &str) -> Option<String> {
    let (mut latin, mut non_latin) = (0usize, 0usize);
    for ch in text.chars() {
        if ('A'..='z').contains(&ch) {
            latin += 1;
        } else if ch >= '\u{300}' {
            // langdetect compares the block id with a block name here, so the
            // 'Latin Extended Additional' exclusion never applies.
            non_latin += 1;
        }
    }
    if latin * 2 < non_latin {
        Some(
            text.chars()
                .filter(|&ch| !('A'..='z').contains(&ch))
                .collect(),
        )
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_like_ngram() {
        assert_eq!(normalize('a'), 'a');
        assert_eq!(normalize('1'), ' ');
        assert_eq!(normalize('\u{219}'), '\u{15f}');
        assert_eq!(normalize('\u{2014}'), ' ');
        assert_eq!(normalize('\u{3044}'), '\u{3042}');
        assert_eq!(normalize('\u{ac01}'), '\u{ac00}');
        assert_eq!(normalize('\u{1ea1}'), '\u{1ec3}');
    }

    #[test]
    fn prepare_removes_urls_mails_and_spaces() {
        assert_eq!(prepare("see https://x.org/a  now", 10000), "see now");
        assert_eq!(prepare("mail a.b@c.de ok", 10000), "mail ok");
        assert_eq!(prepare("abcdef", 3), "abc");
        assert_eq!(prepare("  a", 10000), " a");
        assert_eq!(prepare("abc", -1), "");
    }

    #[test]
    fn vietnamese_combining_marks() {
        assert_eq!(prepare("a\u{301}", 10000), "\u{e1}");
        assert_eq!(prepare("x\u{301}", 10000), "x\u{301}");
    }

    #[test]
    fn cleaning_drops_latin_in_non_latin_text() {
        assert_eq!(clean("abc"), None);
        assert_eq!(clean("ab привет мир"), Some(" привет мир".to_string()));
    }
}
