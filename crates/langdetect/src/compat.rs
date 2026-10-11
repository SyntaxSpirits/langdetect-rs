use std::sync::OnceLock;

use crate::tables::{LATEST_UNICODE, NOT_UPPER_IN, UPPER};

/// How `sum()` adds the probabilities when they are normalised.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SumMode {
    /// Left-to-right addition, as in CPython up to 3.11.
    Sequential,
    /// Neumaier compensated summation, as in CPython 3.12 and later.
    Neumaier,
}

/// Unicode version that decides which characters `str.isupper()` reports as uppercase.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnicodeVersion {
    V13_0,
    V14_0,
    V15_0,
    V15_1,
    V16_0,
    V17_0,
}

const VERSIONS: [(UnicodeVersion, &str); 6] = [
    (UnicodeVersion::V13_0, "13.0.0"),
    (UnicodeVersion::V14_0, "14.0.0"),
    (UnicodeVersion::V15_0, "15.0.0"),
    (UnicodeVersion::V15_1, "15.1.0"),
    (UnicodeVersion::V16_0, "16.0.0"),
    (UnicodeVersion::V17_0, "17.0.0"),
];

impl UnicodeVersion {
    /// Parses `unicodedata.unidata_version`, e.g. `"15.1.0"`.
    pub fn from_unidata(version: &str) -> Option<Self> {
        VERSIONS
            .iter()
            .find(|(_, v)| *v == version)
            .map(|(u, _)| *u)
    }

    pub(crate) fn from_index(i: u8) -> Option<Self> {
        VERSIONS.get(i as usize).map(|(u, _)| *u)
    }

    pub fn as_str(self) -> &'static str {
        VERSIONS[self as usize].1
    }
}

/// The Python behaviour to reproduce. Results are bit-identical to langdetect running on a
/// CPython whose version matches these settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Compat {
    pub sum: SumMode,
    pub unicode: UnicodeVersion,
}

impl Compat {
    /// Settings of CPython `3.minor`; minors outside 9..=15 are clamped.
    pub fn python(minor: u32) -> Self {
        let unicode = match minor {
            0..=10 => UnicodeVersion::V13_0,
            11 => UnicodeVersion::V14_0,
            12 => UnicodeVersion::V15_0,
            13 => UnicodeVersion::V15_1,
            14 => UnicodeVersion::V16_0,
            _ => UnicodeVersion::V17_0,
        };
        let sum = if minor >= 12 {
            SumMode::Neumaier
        } else {
            SumMode::Sequential
        };
        Compat { sum, unicode }
    }
}

impl Default for Compat {
    /// CPython 3.14.
    fn default() -> Self {
        Compat::python(14)
    }
}

pub(crate) struct UpperTable {
    bmp: Box<[u64; 1024]>,
    astral: Vec<u32>,
}

impl UpperTable {
    fn build(version: UnicodeVersion) -> Self {
        let excluded: &[u32] = if version.as_str() == LATEST_UNICODE {
            &[]
        } else {
            NOT_UPPER_IN
                .iter()
                .find(|(v, _)| *v == version.as_str())
                .unwrap()
                .1
        };
        let mut bmp = Box::new([0u64; 1024]);
        let mut astral = Vec::new();
        for &(start, end) in UPPER {
            for cp in start..=end {
                if excluded.contains(&cp) {
                    continue;
                }
                if cp < 0x10000 {
                    bmp[(cp >> 6) as usize] |= 1 << (cp & 63);
                } else {
                    astral.push(cp);
                }
            }
        }
        UpperTable { bmp, astral }
    }

    pub(crate) fn get(version: UnicodeVersion) -> &'static UpperTable {
        static TABLES: [OnceLock<UpperTable>; 6] = [const { OnceLock::new() }; 6];
        TABLES[version as usize].get_or_init(|| UpperTable::build(version))
    }

    #[inline]
    pub(crate) fn is_upper(&self, c: char) -> bool {
        let cp = c as u32;
        if cp < 0x10000 {
            self.bmp[(cp >> 6) as usize] & (1 << (cp & 63)) != 0
        } else {
            self.astral.binary_search(&cp).is_ok()
        }
    }
}

pub(crate) fn sum(prob: &[f64], mode: SumMode) -> f64 {
    let Some((&first, rest)) = prob.split_first() else {
        return 0.0;
    };
    let mut total = first;
    match mode {
        SumMode::Sequential => {
            for &x in rest {
                total += x;
            }
            total
        }
        SumMode::Neumaier => {
            let mut c = 0.0f64;
            for &x in rest {
                let t = total + x;
                let (big, small) = if total.abs() >= x.abs() {
                    (total, x)
                } else {
                    (x, total)
                };
                c += (big - t) + small;
                total = t;
            }
            if c != 0.0 && c.is_finite() {
                total += c;
            }
            total
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neumaier_differs_from_sequential() {
        let xs = [1.0, 1e100, 1.0, -1e100];
        assert_eq!(sum(&xs, SumMode::Sequential), 0.0);
        assert_eq!(sum(&xs, SumMode::Neumaier), 2.0);
    }

    #[test]
    fn uppercase_depends_on_unicode_version() {
        let old = UpperTable::get(UnicodeVersion::V13_0);
        let new = UpperTable::get(UnicodeVersion::V17_0);
        assert!(old.is_upper('A') && new.is_upper('A'));
        assert!(!old.is_upper('a'));
        assert!(!old.is_upper('\u{1C89}') && new.is_upper('\u{1C89}'));
    }

    #[test]
    fn python_versions() {
        assert_eq!(Compat::python(11).sum, SumMode::Sequential);
        assert_eq!(Compat::python(12).sum, SumMode::Neumaier);
        assert_eq!(
            UnicodeVersion::from_unidata("15.1.0"),
            Some(UnicodeVersion::V15_1)
        );
        assert_eq!(UnicodeVersion::from_unidata("12.1.0"), None);
    }
}
