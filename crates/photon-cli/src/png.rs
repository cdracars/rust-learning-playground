//! A from-scratch PNG encoder, including a real DEFLATE compressor.
//!
//! PNG = signature + chunks. Each chunk is `length | type | data | crc32`.
//! The pixels go in `IDAT` chunks, compressed with zlib (DEFLATE plus a
//! small header and an Adler-32 checksum). DEFLATE here uses LZ77 (replace
//! repeated byte runs with back-references) followed by the *fixed* Huffman
//! code from RFC 1951. Before compression every row is "filtered" (e.g.
//! stored as the difference from the row above), which makes smooth
//! gradients compress dramatically better.
//!
//! **Rust concepts:** bit manipulation, `const` lookup tables computed at
//! compile time with `const fn` loops, `impl Write` for generic output,
//! and the `?` operator for `io::Result`.

use std::io::{self, Write};

// ---------------------------------------------------------------------------
// Checksums
// ---------------------------------------------------------------------------

/// CRC-32 lookup table, built by the compiler, not at runtime.
const CRC_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut n = 0;
    while n < 256 {
        // `for` loops aren't allowed in const contexts yet; `while` is.
        let mut c = n as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        table[n] = c;
        n += 1;
    }
    table
};

pub fn crc32(chunks: &[&[u8]]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &bytes in chunks {
        for &b in bytes {
            c = CRC_TABLE[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
        }
    }
    c ^ 0xFFFF_FFFF
}

pub fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65_521;
    let (mut a, mut b) = (1u32, 0u32);
    // Summing 5552 bytes can't overflow a u32, so reduce only that often.
    for chunk in data.chunks(5552) {
        for &x in chunk {
            a += x as u32;
            b += a;
        }
        a %= MOD;
        b %= MOD;
    }
    (b << 16) | a
}

// ---------------------------------------------------------------------------
// DEFLATE (RFC 1951) with fixed Huffman codes
// ---------------------------------------------------------------------------

/// DEFLATE packs bits starting from the least significant bit of each byte.
struct BitWriter {
    out: Vec<u8>,
    acc: u64,
    n: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self { out: Vec::new(), acc: 0, n: 0 }
    }

    fn bits(&mut self, value: u32, count: u32) {
        self.acc |= (value as u64) << self.n;
        self.n += count;
        while self.n >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.n -= 8;
        }
    }

    /// Huffman codes are defined most-significant-bit first, so they're
    /// bit-reversed before being written LSB first.
    fn code(&mut self, code: u32, len: u32) {
        self.bits(code.reverse_bits() >> (32 - len), len);
    }

    fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.out.push(self.acc as u8);
        }
        self.out
    }
}

const LEN_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195,
    227, 258,
];
const LEN_EXTRA: [u8; 29] =
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073,
    4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] =
    [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];

/// Write a literal/length symbol (0..=287) with the fixed Huffman code.
fn write_litlen(w: &mut BitWriter, sym: u32) {
    match sym {
        0..=143 => w.code(0x30 + sym, 8),
        144..=255 => w.code(0x190 + sym - 144, 9),
        256..=279 => w.code(sym - 256, 7),
        _ => w.code(0xC0 + sym - 280, 8),
    }
}

/// Index of the last table entry `<= value` (tables are sorted).
fn bucket(table: &[u16], value: usize) -> usize {
    table.partition_point(|&base| base as usize <= value) - 1
}

const WINDOW: usize = 32 * 1024;
const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 258;
const HASH_BITS: u32 = 15;
const MAX_CHAIN: usize = 48;

/// Compress `data` into a raw DEFLATE stream (a single fixed-Huffman block).
pub fn deflate(data: &[u8]) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.bits(1, 1); // BFINAL: this is the last block
    w.bits(1, 2); // BTYPE = 01: fixed Huffman codes

    // Hash chains: `head[h]` is the latest position whose next 3 bytes hash
    // to `h`; `prev[i % WINDOW]` links to the previous such position.
    let hash = |i: usize| {
        let v = u32::from_le_bytes([data[i], data[i + 1], data[i + 2], 0]);
        (v.wrapping_mul(2_654_435_761) >> (32 - HASH_BITS)) as usize
    };
    let mut head = vec![usize::MAX; 1 << HASH_BITS];
    let mut prev = vec![usize::MAX; WINDOW];
    let insert = |i: usize, head: &mut [usize], prev: &mut [usize]| {
        if i + MIN_MATCH <= data.len() {
            let h = hash(i);
            prev[i % WINDOW] = head[h];
            head[h] = i;
        }
    };

    let mut i = 0;
    while i < data.len() {
        let (mut best_len, mut best_dist) = (0, 0);
        if i + MIN_MATCH <= data.len() {
            let max_len = MAX_MATCH.min(data.len() - i);
            let mut cand = head[hash(i)];
            let mut steps = 0;
            while cand != usize::MAX && i - cand <= WINDOW && steps < MAX_CHAIN {
                let len = data[cand..].iter().zip(&data[i..i + max_len]).take_while(|(a, b)| a == b).count();
                if len > best_len {
                    (best_len, best_dist) = (len, i - cand);
                    if len == max_len {
                        break;
                    }
                }
                let next = prev[cand % WINDOW];
                // Stale entries from an older trip around the ring buffer.
                if next == usize::MAX || next >= cand {
                    break;
                }
                cand = next;
                steps += 1;
            }
        }

        if best_len >= MIN_MATCH {
            let li = bucket(&LEN_BASE, best_len);
            write_litlen(&mut w, 257 + li as u32);
            w.bits((best_len - LEN_BASE[li] as usize) as u32, LEN_EXTRA[li] as u32);
            let di = bucket(&DIST_BASE, best_dist);
            w.code(di as u32, 5);
            w.bits((best_dist - DIST_BASE[di] as usize) as u32, DIST_EXTRA[di] as u32);
            for k in i..i + best_len {
                insert(k, &mut head, &mut prev);
            }
            i += best_len;
        } else {
            write_litlen(&mut w, data[i] as u32);
            insert(i, &mut head, &mut prev);
            i += 1;
        }
    }
    write_litlen(&mut w, 256); // end of block
    w.finish()
}

/// zlib = 2-byte header + DEFLATE stream + big-endian Adler-32 of the input.
pub fn zlib(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01];
    out.extend(deflate(data));
    out.extend(adler32(data).to_be_bytes());
    out
}

// ---------------------------------------------------------------------------
// PNG
// ---------------------------------------------------------------------------

/// Paeth predictor from the PNG spec: guess a byte from its left, upper and
/// upper-left neighbours.
fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let p = a as i16 + b as i16 - c as i16;
    let (pa, pb, pc) = ((p - a as i16).abs(), (p - b as i16).abs(), (p - c as i16).abs());
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Filter every row with each of the five PNG filters and keep whichever
/// produces the smallest values (a standard heuristic for compressibility).
fn filter_rows(rgb: &[u8], width: usize, height: usize) -> Vec<u8> {
    const BPP: usize = 3;
    let stride = width * BPP;
    let zero_row = vec![0u8; stride];
    let mut out = Vec::with_capacity((stride + 1) * height);
    let mut candidate = vec![0u8; stride];

    for y in 0..height {
        let row = &rgb[y * stride..(y + 1) * stride];
        let up = if y > 0 { &rgb[(y - 1) * stride..y * stride] } else { &zero_row[..] };
        let mut best: Option<(u8, u64, Vec<u8>)> = None;

        for filter in 0..5u8 {
            for i in 0..stride {
                let a = if i >= BPP { row[i - BPP] } else { 0 };
                let c = if i >= BPP { up[i - BPP] } else { 0 };
                let b = up[i];
                let predicted = match filter {
                    0 => 0,
                    1 => a,
                    2 => b,
                    3 => ((a as u16 + b as u16) / 2) as u8,
                    _ => paeth(a, b, c),
                };
                candidate[i] = row[i].wrapping_sub(predicted);
            }
            // Treat bytes as signed: small positive *or* negative is good.
            let score: u64 = candidate.iter().map(|&v| (v as i8).unsigned_abs() as u64).sum();
            if best.as_ref().is_none_or(|(_, s, _)| score < *s) {
                best = Some((filter, score, candidate.clone()));
            }
        }
        let (filter, _, bytes) = best.expect("five filters were tried");
        out.push(filter);
        out.extend_from_slice(&bytes);
    }
    out
}

fn chunk(w: &mut impl Write, kind: &[u8; 4], data: &[u8]) -> io::Result<()> {
    w.write_all(&(data.len() as u32).to_be_bytes())?;
    w.write_all(kind)?;
    w.write_all(data)?;
    w.write_all(&crc32(&[kind, data]).to_be_bytes())
}

/// Encode 8-bit RGB pixels (row-major, top row first) as a PNG.
pub fn write_png(w: &mut impl Write, width: usize, height: usize, rgb: &[u8]) -> io::Result<()> {
    assert_eq!(rgb.len(), width * height * 3, "pixel buffer size mismatch");
    w.write_all(b"\x89PNG\r\n\x1a\n")?;

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend((width as u32).to_be_bytes());
    ihdr.extend((height as u32).to_be_bytes());
    ihdr.extend([8, 2, 0, 0, 0]); // 8 bits/channel, truecolor RGB, defaults
    chunk(w, b"IHDR", &ihdr)?;

    chunk(w, b"IDAT", &zlib(&filter_rows(rgb, width, height)))?;
    chunk(w, b"IEND", &[])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_known_value() {
        // The standard check value for CRC-32.
        assert_eq!(crc32(&[b"123456789"]), 0xCBF4_3926);
        assert_eq!(crc32(&[b"1234", b"56789"]), 0xCBF4_3926);
    }

    #[test]
    fn adler32_known_value() {
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    }

    /// A minimal DEFLATE *decoder* for fixed-Huffman blocks, just to prove
    /// the encoder round-trips without pulling in a dependency.
    fn inflate_fixed(stream: &[u8]) -> Vec<u8> {
        let mut pos = 0usize;
        let mut bit = |n: u32| -> u32 {
            let mut v = 0;
            for k in 0..n {
                let b = (stream[pos / 8] >> (pos % 8)) & 1;
                v |= (b as u32) << k;
                pos += 1;
            }
            v
        };
        assert_eq!(bit(1), 1);
        assert_eq!(bit(2), 1);
        let mut out: Vec<u8> = Vec::new();
        loop {
            // Read Huffman code MSB-first, one bit at a time.
            let mut code = 0u32;
            let mut len = 0;
            let sym = loop {
                code = (code << 1) | bit(1);
                len += 1;
                match (len, code) {
                    (7, 0..=0x17) => break 256 + code,
                    (8, 0x30..=0xBF) => break code - 0x30,
                    (8, 0xC0..=0xC7) => break 280 + code - 0xC0,
                    (9, 0x190..=0x1FF) => break 144 + code - 0x190,
                    _ => {}
                }
            };
            match sym {
                0..=255 => out.push(sym as u8),
                256 => return out,
                _ => {
                    let li = (sym - 257) as usize;
                    let len = LEN_BASE[li] as usize + bit(LEN_EXTRA[li] as u32) as usize;
                    let mut d = 0u32;
                    for _ in 0..5 {
                        d = (d << 1) | bit(1);
                    }
                    let di = d as usize;
                    let dist = DIST_BASE[di] as usize + bit(DIST_EXTRA[di] as u32) as usize;
                    let start = out.len() - dist;
                    for k in 0..len {
                        out.push(out[start + k]);
                    }
                }
            }
        }
    }

    #[test]
    fn deflate_round_trips() {
        let mut inputs: Vec<Vec<u8>> = vec![
            vec![],
            b"a".to_vec(),
            b"abcabcabcabcabcabcabc hello hello hello".to_vec(),
            vec![7u8; 100_000],
        ];
        // Pseudo-random bytes with some repetition, longer than the window.
        let mut x = 12345u32;
        inputs.push(
            (0..120_000)
                .map(|i| {
                    x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
                    if i % 1000 < 500 { (x >> 24) as u8 } else { (i % 7) as u8 }
                })
                .collect(),
        );
        for input in inputs {
            assert_eq!(inflate_fixed(&deflate(&input)), input);
        }
    }

    #[test]
    fn compresses_repetitive_data() {
        let data = vec![42u8; 50_000];
        assert!(deflate(&data).len() < 1000);
    }

    #[test]
    fn png_has_valid_structure() {
        let mut buf = Vec::new();
        write_png(&mut buf, 2, 2, &[255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]).unwrap();
        assert_eq!(&buf[..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(&buf[12..16], b"IHDR");
        assert_eq!(&buf[buf.len() - 8..buf.len() - 4], b"IEND");
    }
}
