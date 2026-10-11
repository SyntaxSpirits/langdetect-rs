use std::cell::RefCell;
use std::sync::Arc;

const N: usize = 624;
const M: usize = 397;

/// Seed of the random walk, mirroring `DetectorFactory.seed`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Seed {
    /// `seed = None`: a fresh random seed for every detection.
    #[default]
    Os,
    /// An integer seed, as `random.Random(n)` (the sign is ignored, as in CPython).
    Int(u128),
    /// A Mersenne Twister state as returned by `random.Random(seed).getstate()[1]`
    /// (624 words followed by the position).
    State(Arc<[u32; N + 1]>),
}

impl Seed {
    pub fn int(n: i128) -> Self {
        Seed::Int(n.unsigned_abs())
    }

    /// Builds a seed from the 625 integers of `random.getstate()[1]`.
    pub fn from_state(state: &[u32]) -> Option<Self> {
        let arr: [u32; N + 1] = state.try_into().ok()?;
        if arr[N] as usize > N {
            return None;
        }
        Some(Seed::State(Arc::new(arr)))
    }

    pub(crate) fn rng(&self) -> PyRandom {
        match self {
            Seed::Os => PyRandom::from_os(),
            Seed::Int(n) => {
                let mut key = Vec::with_capacity(4);
                let mut v = *n;
                loop {
                    key.push(v as u32);
                    v >>= 32;
                    if v == 0 {
                        break;
                    }
                }
                PyRandom::from_key(&key)
            }
            Seed::State(s) => {
                let mut mt = [0u32; N];
                mt.copy_from_slice(&s[..N]);
                PyRandom {
                    mt,
                    idx: s[N] as usize,
                    gauss_next: None,
                }
            }
        }
    }
}

/// CPython's `random.Random`: MT19937 with `init_by_array` seeding, `random()`, `gauss()` and
/// `_randbelow()`, reproduced exactly.
pub(crate) struct PyRandom {
    mt: [u32; N],
    idx: usize,
    gauss_next: Option<f64>,
}

impl PyRandom {
    fn init_genrand(s: u32) -> [u32; N] {
        let mut mt = [0u32; N];
        mt[0] = s;
        for i in 1..N {
            mt[i] = 1812433253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        mt
    }

    pub(crate) fn from_key(key: &[u32]) -> Self {
        let mut mt = Self::init_genrand(19650218);
        let (mut i, mut j) = (1usize, 0usize);
        let mut k = N.max(key.len());
        while k > 0 {
            mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1664525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        k = N - 1;
        while k > 0 {
            mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1566083941))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        mt[0] = 0x80000000;
        PyRandom {
            mt,
            idx: N,
            gauss_next: None,
        }
    }

    /// A fresh seed per call, like `random.seed(None)`. Instead of reading 2.5 KB from the
    /// operating system each time, the 624 key words come from a per-thread xoshiro256**
    /// generator seeded by the operating system and reseeded in a forked child.
    fn from_os() -> Self {
        thread_local! {
            static ENTROPY: RefCell<Option<(u32, [u64; 4])>> = const { RefCell::new(None) };
        }
        let pid = std::process::id();
        let mut key = [0u32; N];
        ENTROPY.with_borrow_mut(|entropy| {
            if !matches!(entropy, Some((p, _)) if *p == pid) {
                let mut bytes = [0u8; 32];
                getrandom::fill(&mut bytes).expect("operating system random source failed");
                let mut s = [0u64; 4];
                for (w, c) in s.iter_mut().zip(bytes.chunks_exact(8)) {
                    *w = u64::from_le_bytes(c.try_into().unwrap());
                }
                if s == [0; 4] {
                    s[0] = 1;
                }
                *entropy = Some((pid, s));
            }
            let s = &mut entropy.as_mut().unwrap().1;
            for pair in key.chunks_exact_mut(2) {
                let r = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
                let t = s[1] << 17;
                s[2] ^= s[0];
                s[3] ^= s[1];
                s[1] ^= s[2];
                s[0] ^= s[3];
                s[2] ^= t;
                s[3] = s[3].rotate_left(45);
                pair[0] = r as u32;
                pair[1] = (r >> 32) as u32;
            }
        });
        Self::from_key(&key)
    }

    fn twist(&mut self) {
        #[inline(always)]
        fn mix(upper: u32, lower: u32, far: u32) -> u32 {
            let y = (upper & 0x80000000) | (lower & 0x7fffffff);
            far ^ (y >> 1) ^ ((y & 1).wrapping_neg() & 0x9908b0df)
        }
        let mt = &mut self.mt;
        for kk in 0..N - M {
            mt[kk] = mix(mt[kk], mt[kk + 1], mt[kk + M]);
        }
        for kk in N - M..N - 1 {
            mt[kk] = mix(mt[kk], mt[kk + 1], mt[kk + M - N]);
        }
        mt[N - 1] = mix(mt[N - 1], mt[0], mt[M - 1]);
        self.idx = 0;
    }

    fn next_u32(&mut self) -> u32 {
        if self.idx >= N {
            self.twist();
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }

    fn random(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }

    /// `random.gauss(0.0, 1.0)`.
    pub(crate) fn gauss(&mut self) -> f64 {
        if let Some(z) = self.gauss_next.take() {
            return z;
        }
        let x2pi = self.random() * (2.0 * std::f64::consts::PI);
        let g2rad = (-2.0 * (1.0 - self.random()).ln()).sqrt();
        self.gauss_next = Some(x2pi.sin() * g2rad);
        x2pi.cos() * g2rad
    }

    /// `random._randbelow(n)` for `1 <= n < 2**32`.
    pub(crate) fn randbelow(&mut self, n: usize) -> usize {
        let k = usize::BITS - n.leading_zeros();
        debug_assert!((1..=32).contains(&k));
        loop {
            let r = (self.next_u32() >> (32 - k)) as usize;
            if r < n {
                return r;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Values from CPython: r = random.Random(seed); r.getstate()[1]; r.gauss(0.0, 1.0);
    // r.getrandbits(17); r.random().
    fn check(seed: u128, s0: u32, s1: u32, s623: u32, g: f64, bits: u32, x: f64) {
        let mut r = Seed::Int(seed).rng();
        assert_eq!((r.mt[0], r.mt[1], r.mt[623]), (s0, s1, s623));
        assert_eq!(r.gauss().to_bits(), g.to_bits());
        assert_eq!(r.next_u32() >> 15, bits);
        assert_eq!(r.random().to_bits(), x.to_bits());
    }

    #[test]
    fn matches_cpython() {
        check(
            0,
            2147483648,
            766982754,
            2902720905,
            0.9417154046806644,
            55125,
            0.04048437818077755,
        );
        check(
            12345,
            2147483648,
            2105189241,
            238504783,
            -0.12380079558885389,
            108161,
            0.802848393459915,
        );
        check(
            0x123456789abcdef0,
            2147483648,
            3823438522,
            3246624679,
            1.5092144802014515,
            116145,
            0.4749266540075584,
        );
    }

    #[test]
    fn gauss_caches_second_value() {
        let mut a = Seed::Int(7).rng();
        let first = a.gauss();
        let second = a.gauss();
        assert_ne!(first.to_bits(), second.to_bits());
        let mut b = Seed::Int(7).rng();
        assert_eq!(b.gauss().to_bits(), first.to_bits());
    }

    #[test]
    fn negative_seed_is_abs() {
        assert_eq!(Seed::int(-5), Seed::Int(5));
    }
}
