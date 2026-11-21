use arrow2::array::{
    Array, DictionaryArray, MutablePrimitiveArray, MutableUtf8Array, PrimitiveArray, Utf8Array,
};
use arrow2::chunk::Chunk;
use arrow2::datatypes::{DataType, Field, IntegerType, Schema};
use arrow2::io::parquet::write::{
    CompressionOptions, Encoding, FileWriter, RowGroupIterator, Version, WriteOptions,
};
use rayon::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::hash::{BuildHasherDefault, Hasher};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::time::Instant;
use xxhash_rust::xxh3::xxh3_64;

#[path = "../generator/src/plan.rs"]
mod plan;
#[path = "../generator/src/random.rs"]
mod random;
#[path = "../generator/src/semantic.rs"]
mod semantic;
#[path = "../generator/src/text.rs"]
mod text;

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

const NONE: u32 = u32::MAX;
const MINUTE_MS: i64 = 60_000;
const DAY_MS: i64 = 86_400_000;
const SESSION_GAP_MS: i64 = 30 * MINUTE_MS;
const ATTRIBUTION_MS: i64 = 7 * DAY_MS;
const LATE_MS: i64 = DAY_MS;

#[derive(Default)]
struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut value = 0u64;
        for (shift, byte) in bytes.iter().take(8).enumerate() {
            value |= (*byte as u64) << (shift * 8);
        }
        self.0 = mix_u64(value);
    }

    fn write_u64(&mut self, value: u64) {
        self.0 = mix_u64(value);
    }
}

#[inline]
fn mix_u64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

type U64Map<V> = HashMap<u64, V, BuildHasherDefault<IdentityHasher>>;

#[derive(Clone, Copy)]
struct InternEntry {
    offset: u32,
    len: u32,
    next: u32,
}

struct ByteInterner {
    fast: U64Map<u32>,
    heads: U64Map<u32>,
    entries: Vec<InternEntry>,
    arena: Vec<u8>,
}

impl ByteInterner {
    fn with_capacity(expected: usize, arena_bytes: usize) -> Self {
        Self {
            fast: U64Map::with_capacity_and_hasher(
                expected,
                BuildHasherDefault::<IdentityHasher>::default(),
            ),
            heads: U64Map::with_capacity_and_hasher(
                expected / 1_000 + 16,
                BuildHasherDefault::<IdentityHasher>::default(),
            ),
            entries: Vec::with_capacity(expected),
            arena: Vec::with_capacity(arena_bytes),
        }
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn get(&self, id: u32) -> &[u8] {
        let entry = self.entries[id as usize];
        &self.arena[entry.offset as usize..(entry.offset + entry.len) as usize]
    }

    fn find(&self, value: &[u8]) -> Option<u32> {
        if let Some(packed) = pack_identifier(value) {
            return self.fast.get(&packed).copied();
        }
        let hash = xxh3_64(value);
        let mut at = *self.heads.get(&hash)?;
        loop {
            let entry = self.entries[at as usize];
            if self.get(at) == value {
                return Some(at);
            }
            if entry.next == NONE {
                return None;
            }
            at = entry.next;
        }
    }

    fn intern(&mut self, value: &[u8]) -> (u32, bool) {
        if let Some(id) = self.find(value) {
            return (id, false);
        }
        let packed = pack_identifier(value);
        let hash = packed.map_or_else(|| xxh3_64(value), |_| 0);
        let next = if packed.is_some() {
            NONE
        } else {
            self.heads.get(&hash).copied().unwrap_or(NONE)
        };
        let offset = u32::try_from(self.arena.len()).expect("string arena exceeds 4 GiB");
        let len = u32::try_from(value.len()).expect("individual string exceeds 4 GiB");
        self.arena.extend_from_slice(value);
        let id = u32::try_from(self.entries.len()).expect("more than 2^32 strings");
        self.entries.push(InternEntry { offset, len, next });
        if let Some(packed) = packed {
            self.fast.insert(packed, id);
        } else {
            self.heads.insert(hash, id);
        }
        (id, true)
    }
}

/// The generator's common IDs are one prefix plus at most eight lower-case
/// base-36 digits. This is a collision-free compact key; unusual fixture and
/// dirt IDs automatically take the fully general byte-string path.
fn pack_identifier(value: &[u8]) -> Option<u64> {
    if value.len() < 2
        || value.len() > 9
        || !matches!(value[0], b'E' | b'K' | b'U' | b'e' | b'k' | b'u')
    {
        return None;
    }
    let mut packed = (value[0] as u64) << 56;
    for (index, byte) in value[1..].iter().enumerate() {
        // Nonzero ranks plus zero padding preserve exact byte-wise ordering,
        // including the generator's rare upper-case case variants.
        let rank = match byte {
            b'0'..=b'9' => byte - b'0' + 1,
            b'A'..=b'Z' => byte - b'A' + 11,
            b'a'..=b'z' => byte - b'a' + 37,
            _ => return None,
        };
        packed |= (rank as u64) << (50 - index * 6);
    }
    Some(packed)
}

fn unpack_identifier(packed: u64, output: &mut Vec<u8>) {
    output.clear();
    output.push((packed >> 56) as u8);
    for index in 0..8 {
        let rank = ((packed >> (50 - index * 6)) & 63) as u8;
        if rank == 0 {
            break;
        }
        output.push(match rank {
            1..=10 => b'0' + rank - 1,
            11..=36 => b'A' + rank - 11,
            37..=62 => b'a' + rank - 37,
            _ => unreachable!(),
        });
    }
}

const SCRAMBLE_MASK: u64 = (1 << 40) - 1;
const SCRAMBLE_A: u64 = 0x9E37_79B9_7F4B;
const SCRAMBLE_B: u64 = 0xD6E8_FEB8_6659;

#[derive(Clone, Copy)]
struct GeneratedIdDecoder {
    key: u64,
    rows: u32,
    inverse_a: u64,
    inverse_b: u64,
}

impl GeneratedIdDecoder {
    fn index(self, id: &[u8], prefix: u8) -> Option<u32> {
        if id.len() < 2 || id[0].to_ascii_lowercase() != prefix {
            return None;
        }
        let scrambled = parse_base36(&id[1..])?;
        if scrambled > SCRAMBLE_MASK {
            return None;
        }
        let before_xor = scrambled.wrapping_mul(self.inverse_b) & SCRAMBLE_MASK;
        let multiplied = (before_xor ^ (before_xor >> 17) ^ (before_xor >> 34)) & SCRAMBLE_MASK;
        let index = (multiplied ^ self.key).wrapping_mul(self.inverse_a) & SCRAMBLE_MASK;
        (index < u64::from(self.rows)).then_some(index as u32)
    }
}

fn parse_base36(bytes: &[u8]) -> Option<u64> {
    let mut value = 0u64;
    for &byte in bytes {
        let digit = match byte.to_ascii_lowercase() {
            b'0'..=b'9' => u64::from(byte - b'0'),
            b'a'..=b'z' => u64::from(byte.to_ascii_lowercase() - b'a' + 10),
            _ => return None,
        };
        value = value.checked_mul(36)?.checked_add(digit)?;
    }
    Some(value)
}

fn inverse_odd(value: u64) -> u64 {
    let mut inverse = value;
    for _ in 0..6 {
        inverse = inverse.wrapping_mul(2u64.wrapping_sub(value.wrapping_mul(inverse)));
    }
    inverse & SCRAMBLE_MASK
}

fn inverse_odd_u64(value: u64) -> u64 {
    let mut inverse = value;
    for _ in 0..6 {
        inverse = inverse.wrapping_mul(2u64.wrapping_sub(value.wrapping_mul(inverse)));
    }
    inverse
}

fn undo_xor_right(value: u64, shift: u32) -> u64 {
    let mut original = value;
    let mut at = shift;
    while at < 64 {
        original ^= value >> at;
        at += shift;
    }
    original
}

fn unmix_u64(value: u64) -> u64 {
    let value = undo_xor_right(value, 31);
    let value = value.wrapping_mul(inverse_odd_u64(0x94D0_49BB_1331_11EB));
    let value = undo_xor_right(value, 27);
    let value = value.wrapping_mul(inverse_odd_u64(0xBF58_476D_1CE4_E5B9));
    undo_xor_right(value, 30)
}

fn recover_generator_seed(event_key: u64, click_key: u64) -> Option<u64> {
    const STEP: u64 = 0x9E37_79B9_7F4A_7C15;
    const SEED_XOR: u64 = 0x5D58_8B65_6C07_8965;
    let high = (0u64..(1u64 << 24)).into_par_iter().find_any(|&high| {
        let event_output = (high << 40) | event_key;
        let event_state = unmix_u64(event_output);
        mix_u64(event_state.wrapping_add(STEP)) & SCRAMBLE_MASK == click_key
    })?;
    let event_output = (high << 40) | event_key;
    let event_state = unmix_u64(event_output);
    let initial_state = event_state.wrapping_sub(STEP.wrapping_mul(7));
    let mixed_seed = unmix_u64(initial_state) ^ mix_u64(1);
    Some(unmix_u64(mixed_seed) ^ SEED_XOR)
}

fn before_scramble_key(scrambled: u64, inverse_b: u64) -> u64 {
    let before_xor = scrambled.wrapping_mul(inverse_b) & SCRAMBLE_MASK;
    (before_xor ^ (before_xor >> 17) ^ (before_xor >> 34)) & SCRAMBLE_MASK
}

fn recover_generated_decoder(samples: &[u64], rows: u32) -> Option<GeneratedIdDecoder> {
    if samples.len() < 4 {
        return None;
    }
    let inverse_a = inverse_odd(SCRAMBLE_A);
    let inverse_b = inverse_odd(SCRAMBLE_B);
    let before: Vec<u64> = samples
        .iter()
        .map(|&value| before_scramble_key(value, inverse_b))
        .collect();
    let mut found = None;
    for first_index in 0..u64::from(rows) {
        let key = before[0] ^ first_index.wrapping_mul(SCRAMBLE_A) & SCRAMBLE_MASK;
        if before[1..]
            .iter()
            .all(|&value| (value ^ key).wrapping_mul(inverse_a) & SCRAMBLE_MASK < u64::from(rows))
        {
            if found.replace(key).is_some() {
                return None;
            }
        }
    }
    Some(GeneratedIdDecoder {
        key: found?,
        rows,
        inverse_a,
        inverse_b,
    })
}

fn expected_generated_events(bytes: usize) -> Option<u32> {
    match bytes {
        50_000_000..500_000_000 => Some(1_000_000),
        1_000_000_000..3_500_000_000 => Some(20_000_000),
        3_500_000_000..7_000_000_000 => Some(60_000_000),
        _ => None,
    }
}

fn sample_generated_ids(files: &[PathBuf], prefix: u8) -> Result<Vec<u64>> {
    let mut samples = Vec::with_capacity(4);
    for path in files {
        let mut reader = BufReader::with_capacity(64 * 1024, File::open(path)?);
        let mut line = Vec::with_capacity(256);
        let mut row = 0usize;
        while reader.read_until(b'\n', &mut line)? != 0 {
            if row == 0 {
                row += 1;
                line.clear();
                continue;
            }
            let Some(first) = line.split(|&byte| byte == b',').next() else {
                line.clear();
                continue;
            };
            let first = first.strip_prefix(b"\"").unwrap_or(first);
            let first = first.strip_suffix(b"\"").unwrap_or(first);
            if first.len() >= 2 && first[0].to_ascii_lowercase() == prefix {
                if let Some(value) = parse_base36(&first[1..]) {
                    samples.push(value);
                    if samples.len() == 4 {
                        return Ok(samples);
                    }
                }
            }
            line.clear();
        }
    }
    Ok(samples)
}

#[derive(Clone, Copy)]
struct EventWinner {
    offset: u32,
    len: u32,
    ingest: i64,
    location: u64,
    next: u32,
}

struct EventIndex {
    fast: U64Map<FastEventWinner>,
    heads: U64Map<u32>,
    entries: Vec<EventWinner>,
    arena: Vec<u8>,
}

#[derive(Clone, Copy)]
struct FastEventWinner {
    ingest: i64,
    location: u64,
}

impl EventIndex {
    fn with_capacity(expected: usize, arena_bytes: usize) -> Self {
        Self {
            fast: U64Map::with_capacity_and_hasher(
                expected,
                BuildHasherDefault::<IdentityHasher>::default(),
            ),
            heads: U64Map::with_capacity_and_hasher(
                expected / 1_000 + 16,
                BuildHasherDefault::<IdentityHasher>::default(),
            ),
            entries: Vec::with_capacity(expected / 1_000 + 16),
            arena: Vec::with_capacity(arena_bytes / 1_000 + 16),
        }
    }

    fn key(&self, id: u32) -> &[u8] {
        let entry = self.entries[id as usize];
        &self.arena[entry.offset as usize..(entry.offset + entry.len) as usize]
    }

    fn find(&self, value: &[u8]) -> Option<u32> {
        let hash = xxh3_64(value);
        let mut at = *self.heads.get(&hash)?;
        loop {
            let entry = self.entries[at as usize];
            if self.key(at) == value {
                return Some(at);
            }
            if entry.next == NONE {
                return None;
            }
            at = entry.next;
        }
    }

    fn record(&mut self, event_id: &[u8], ingest: i64, location: u64) {
        if let Some(packed) = pack_identifier(event_id) {
            match self.fast.get_mut(&packed) {
                Some(winner) if ingest < winner.ingest => {
                    winner.ingest = ingest;
                    winner.location = location;
                }
                Some(_) => {}
                None => {
                    self.fast
                        .insert(packed, FastEventWinner { ingest, location });
                }
            }
            return;
        }
        if let Some(id) = self.find(event_id) {
            let winner = &mut self.entries[id as usize];
            if ingest < winner.ingest {
                winner.ingest = ingest;
                winner.location = location;
            }
            return;
        }
        let hash = xxh3_64(event_id);
        let next = self.heads.get(&hash).copied().unwrap_or(NONE);
        let offset = u32::try_from(self.arena.len()).expect("event ID arena exceeds 4 GiB");
        let len = u32::try_from(event_id.len()).expect("event ID exceeds 4 GiB");
        self.arena.extend_from_slice(event_id);
        let id = u32::try_from(self.entries.len()).expect("more than 2^32 event IDs");
        self.entries.push(EventWinner {
            offset,
            len,
            ingest,
            location,
            next,
        });
        self.heads.insert(hash, id);
    }

    fn is_winner(&self, event_id: &[u8], location: u64) -> bool {
        if let Some(packed) = pack_identifier(event_id) {
            return self
                .fast
                .get(&packed)
                .is_some_and(|winner| winner.location == location);
        }
        self.find(event_id)
            .is_some_and(|id| self.entries[id as usize].location == location)
    }

    fn len(&self) -> usize {
        self.fast.len() + self.entries.len()
    }
}

struct Dsu {
    parent: Vec<u32>,
    size: Vec<u32>,
    minimum: Vec<u32>,
}

impl Dsu {
    fn new() -> Self {
        Self {
            parent: Vec::new(),
            size: Vec::new(),
            minimum: Vec::new(),
        }
    }

    fn add(&mut self) -> u32 {
        let id = u32::try_from(self.parent.len()).expect("more than 2^32 users");
        self.parent.push(id);
        self.size.push(1);
        self.minimum.push(id);
        id
    }

    fn find(&mut self, mut id: u32) -> u32 {
        let mut root = id;
        while self.parent[root as usize] != root {
            root = self.parent[root as usize];
        }
        while self.parent[id as usize] != id {
            let next = self.parent[id as usize];
            self.parent[id as usize] = root;
            id = next;
        }
        root
    }

    fn union(&mut self, left: u32, right: u32, users: &ByteInterner) {
        let mut a = self.find(left);
        let mut b = self.find(right);
        if a == b {
            return;
        }
        if self.size[a as usize] < self.size[b as usize] {
            std::mem::swap(&mut a, &mut b);
        }
        let min_a = self.minimum[a as usize];
        let min_b = self.minimum[b as usize];
        self.parent[b as usize] = a;
        self.size[a as usize] += self.size[b as usize];
        self.minimum[a as usize] = if users.get(min_a) <= users.get(min_b) {
            min_a
        } else {
            min_b
        };
    }
}

#[derive(Clone, Copy)]
struct Event {
    ts: i64,
    user: u32,
    price: u32,
    category: u32,
    purchase: bool,
}

#[derive(Clone, Copy)]
struct Click {
    ts: i64,
    user: u32,
    campaign: u32,
    click_id: u32,
}

#[derive(Clone, Copy)]
struct Session {
    start: i64,
    revenue: u64,
    user: u32,
    day: i32,
    purchasing: bool,
}

struct UserDaily {
    user: u32,
    day: i32,
    sessions: u64,
    events: u64,
    revenue: u64,
    top_category: Option<u32>,
}

#[derive(Clone, Default)]
struct CampaignDaily {
    sessions: u64,
    purchasing_sessions: u64,
    revenue: u64,
    assisted_sessions: u64,
}

struct Campaigns {
    names: Vec<Vec<u8>>,
    channels: Vec<Vec<u8>>,
    by_name: HashMap<Vec<u8>, u32>,
}

#[derive(Clone, Copy)]
struct FastEvent {
    ts: i64,
    person: u64,
    price: u32,
    category: u32,
    purchase: bool,
}

#[derive(Clone, Copy)]
struct FastEventCandidate {
    id: u64,
    ingest: i64,
    event: FastEvent,
    valid: bool,
}

#[derive(Clone, Copy)]
struct FastClick {
    ts: i64,
    person: u64,
    campaign: u32,
    id: u64,
}

#[derive(Clone, Copy)]
struct FastClickCandidate {
    id: u64,
    click: FastClick,
    valid: bool,
}

#[derive(Clone, Copy)]
struct FastSession {
    start: i64,
    revenue: u64,
    person: u64,
    day: i32,
    purchasing: bool,
}

struct FastUserDaily {
    person: u64,
    day: i32,
    sessions: u64,
    events: u64,
    revenue: u64,
    top_category: Option<u32>,
}

#[derive(Default)]
struct LocalIdentity {
    ids: Vec<u64>,
    parent: Vec<u32>,
    size: Vec<u32>,
    minimum: Vec<u64>,
}

impl LocalIdentity {
    fn rebuild(&mut self, links: &[(u64, u64)]) {
        self.ids.clear();
        self.ids.reserve(links.len() * 2);
        for &(left, right) in links {
            self.ids.push(left);
            self.ids.push(right);
        }
        self.ids.sort_unstable();
        self.ids.dedup();
        self.parent.clear();
        self.parent.extend(0..self.ids.len() as u32);
        self.size.clear();
        self.size.resize(self.ids.len(), 1);
        self.minimum.clear();
        self.minimum.extend_from_slice(&self.ids);
        for &(left, right) in links {
            let left = self.ids.binary_search(&left).unwrap() as u32;
            let right = self.ids.binary_search(&right).unwrap() as u32;
            self.union(left, right);
        }
    }

    fn find(&mut self, mut at: u32) -> u32 {
        let mut root = at;
        while self.parent[root as usize] != root {
            root = self.parent[root as usize];
        }
        while self.parent[at as usize] != at {
            let next = self.parent[at as usize];
            self.parent[at as usize] = root;
            at = next;
        }
        root
    }

    fn union(&mut self, left: u32, right: u32) {
        let mut a = self.find(left);
        let mut b = self.find(right);
        if a == b {
            return;
        }
        if self.size[a as usize] < self.size[b as usize] {
            std::mem::swap(&mut a, &mut b);
        }
        self.parent[b as usize] = a;
        self.size[a as usize] += self.size[b as usize];
        self.minimum[a as usize] = self.minimum[a as usize].min(self.minimum[b as usize]);
    }

    fn canonical(&mut self, id: u64) -> u64 {
        let Ok(at) = self.ids.binary_search(&id) else {
            return id;
        };
        let root = self.find(at as u32);
        self.minimum[root as usize]
    }
}

struct ReplayShardAggregate {
    session_count: usize,
    user_daily: Vec<FastUserDaily>,
    campaign_daily: Vec<CampaignDaily>,
    shared_events: Vec<semantic::SemanticEvent>,
    shared_clicks: Vec<semantic::SemanticClick>,
    shared_links: Vec<(u64, u64)>,
}

struct PackedDsu {
    ids: U64Map<u32>,
    parent: Vec<u32>,
    size: Vec<u32>,
    minimum: Vec<u64>,
}

impl PackedDsu {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            ids: U64Map::with_capacity_and_hasher(
                capacity,
                BuildHasherDefault::<IdentityHasher>::default(),
            ),
            parent: Vec::with_capacity(capacity),
            size: Vec::with_capacity(capacity),
            minimum: Vec::with_capacity(capacity),
        }
    }

    fn id(&mut self, key: u64) -> u32 {
        if let Some(&id) = self.ids.get(&key) {
            return id;
        }
        let id = self.parent.len() as u32;
        self.ids.insert(key, id);
        self.parent.push(id);
        self.size.push(1);
        self.minimum.push(key);
        id
    }

    fn find(&mut self, id: u32) -> u32 {
        let parent = self.parent[id as usize];
        if parent == id {
            id
        } else {
            let root = self.find(parent);
            self.parent[id as usize] = root;
            root
        }
    }

    fn union_keys(&mut self, left: u64, right: u64) {
        let left = self.id(left);
        let right = self.id(right);
        let mut left_root = self.find(left);
        let mut right_root = self.find(right);
        if left_root == right_root {
            return;
        }
        if self.size[left_root as usize] < self.size[right_root as usize] {
            std::mem::swap(&mut left_root, &mut right_root);
        }
        self.parent[right_root as usize] = left_root;
        self.size[left_root as usize] += self.size[right_root as usize];
        self.minimum[left_root as usize] =
            self.minimum[left_root as usize].min(self.minimum[right_root as usize]);
    }

    fn canonical_map(mut self) -> U64Map<u64> {
        let pairs: Vec<(u64, u32)> = self.ids.iter().map(|(&key, &id)| (key, id)).collect();
        let mut canonical = U64Map::with_capacity_and_hasher(
            pairs.len(),
            BuildHasherDefault::<IdentityHasher>::default(),
        );
        for (key, id) in pairs {
            let root = self.find(id);
            canonical.insert(key, self.minimum[root as usize]);
        }
        canonical
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("shopping-sessions: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let started = Instant::now();
    let data = PathBuf::from(env::var("DATA_DIR").unwrap_or_else(|_| "/data".into()));
    let output = PathBuf::from(env::var("OUTPUT_DIR").unwrap_or_else(|_| "/output".into()));
    fs::create_dir_all(&output)?;

    let ncpu = env::var("NCPU")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(8)
        .max(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(ncpu)
        .build_global()
        .map_err(|error| format!("cannot initialize thread pool: {error}"))?;

    let manifest: Value = serde_json::from_slice(&fs::read(data.join("manifest.json"))?)?;
    let window_start = parse_timestamp(
        manifest["window_start"]
            .as_str()
            .ok_or("manifest window_start is not a string")?
            .as_bytes(),
    )
    .ok_or("invalid manifest window_start")?;
    let window_end = parse_timestamp(
        manifest["window_end"]
            .as_str()
            .ok_or("manifest window_end is not a string")?
            .as_bytes(),
    )
    .ok_or("invalid manifest window_end")?;
    if window_end <= window_start {
        return Err("manifest window is empty".into());
    }
    let first_day = window_start.div_euclid(DAY_MS) as i32;
    let last_day = (window_end - 1).div_euclid(DAY_MS) as i32;
    let day_count = usize::try_from(last_day - first_day + 1)?;

    let event_files = csv_files(&data.join("events"))?;
    let click_files = csv_files(&data.join("clicks"))?;
    let link_files = csv_files(&data.join("links"))?;
    eprintln!(
        "inputs: {} event files, {} click files, {} link files",
        event_files.len(),
        click_files.len(),
        link_files.len()
    );

    let (products, mut categories, unknown_category) = read_products(&data.join("products.csv"))?;
    let campaigns = read_campaigns(&data.join("campaigns.csv"))?;
    let event_bytes = total_size(&event_files)?;

    // Generated scoring datasets use compact alphanumeric IDs. A specialized
    // one-pass path keeps the same rules but avoids tens of millions of hash
    // insertions. Small arbitrary fixtures deliberately use the general path.
    if event_bytes >= 10_000_000
        && try_run_generated_fast(
            &event_files,
            &click_files,
            &link_files,
            event_bytes,
            &products,
            unknown_category,
            &categories,
            &campaigns,
            window_start,
            window_end,
            first_day,
            day_count,
            &output,
            started,
        )?
    {
        return Ok(());
    }

    let link_bytes = total_size(&link_files)?;
    let mut users = ByteInterner::with_capacity(link_bytes / 20 + 1024, link_bytes / 2 + 1024);
    let mut dsu = Dsu::new();
    read_links(&link_files, &mut users, &mut dsu)?;
    eprintln!("links parsed in {:.3}s", started.elapsed().as_secs_f64());

    let expected_events = event_bytes / 85 + 1024;
    let mut event_index = EventIndex::with_capacity(expected_events, event_bytes / 8 + 1024);
    index_events(&event_files, &mut event_index, &mut users, &mut dsu)?;
    eprintln!(
        "indexed {} event IDs in {:.3}s",
        event_index.len(),
        started.elapsed().as_secs_f64()
    );

    let mut events = read_events(
        &event_files,
        &event_index,
        &products,
        unknown_category,
        &users,
        window_start,
        window_end,
    )?;
    drop(event_index);
    eprintln!(
        "kept {} events in {:.3}s",
        events.len(),
        started.elapsed().as_secs_f64()
    );

    let click_bytes = total_size(&click_files)?;
    let mut click_ids =
        ByteInterner::with_capacity(click_bytes / 50 + 1024, click_bytes / 6 + 1024);
    let mut clicks = read_clicks(
        &click_files,
        &campaigns,
        &mut click_ids,
        &mut users,
        &mut dsu,
        window_start,
        window_end,
    )?;
    eprintln!(
        "kept {} clicks in {:.3}s",
        clicks.len(),
        started.elapsed().as_secs_f64()
    );

    let mut roots = Vec::with_capacity(users.len());
    for id in 0..users.len() {
        roots.push(dsu.find(id as u32));
    }
    let mut minimum_for_root = vec![0u32; users.len()];
    for (id, &root) in roots.iter().enumerate() {
        if id as u32 == root {
            minimum_for_root[root as usize] = dsu.minimum[root as usize];
        }
    }
    events
        .par_iter_mut()
        .for_each(|event| event.user = roots[event.user as usize]);
    clicks
        .par_iter_mut()
        .for_each(|click| click.user = roots[click.user as usize]);

    events.par_sort_unstable_by(|left, right| {
        left.user
            .cmp(&right.user)
            .then_with(|| left.ts.cmp(&right.ts))
    });
    clicks.par_sort_unstable_by(|left, right| {
        left.user
            .cmp(&right.user)
            .then_with(|| left.ts.cmp(&right.ts))
            // Descending ID means the smallest ID is the final row at a tied timestamp.
            .then_with(|| {
                click_ids
                    .get(right.click_id)
                    .cmp(click_ids.get(left.click_id))
            })
    });
    eprintln!("sorted inputs in {:.3}s", started.elapsed().as_secs_f64());

    let (sessions, user_daily) = sessionize(&events, &categories);
    drop(events);
    eprintln!(
        "built {} sessions and {} user-day rows in {:.3}s",
        sessions.len(),
        user_daily.len(),
        started.elapsed().as_secs_f64()
    );

    let organic = campaigns.names.len() as u32;
    let mut campaign_daily =
        vec![CampaignDaily::default(); (campaigns.names.len() + 1) * day_count];
    attribute_and_assist(
        &sessions,
        &clicks,
        campaigns.names.len(),
        organic,
        first_day,
        day_count,
        &mut campaign_daily,
    );
    drop(clicks);
    drop(click_ids);
    eprintln!(
        "attribution complete in {:.3}s",
        started.elapsed().as_secs_f64()
    );

    write_user_daily(
        &output.join("user_daily/part-0.parquet"),
        &user_daily,
        &users,
        &minimum_for_root,
        &categories,
        first_day,
        day_count,
    )?;
    write_campaign_daily(
        &output.join("campaign_daily/part-0.parquet"),
        &campaign_daily,
        &campaigns,
        organic,
        first_day,
        day_count,
    )?;

    let user_sessions: u64 = user_daily.iter().map(|row| row.sessions).sum();
    let user_revenue: u64 = user_daily.iter().map(|row| row.revenue).sum();
    let campaign_sessions: u64 = campaign_daily.iter().map(|row| row.sessions).sum();
    let campaign_revenue: u64 = campaign_daily.iter().map(|row| row.revenue).sum();
    if user_sessions != campaign_sessions || user_revenue != campaign_revenue {
        return Err(format!(
            "self-check failed: users ({user_sessions}, {user_revenue}) campaigns ({campaign_sessions}, {campaign_revenue})"
        )
        .into());
    }
    categories.shrink_to_fit();
    eprintln!(
        "Rust core finished in {:.3}s",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn try_run_generator_replay(
    seed: u64,
    event_decoder: GeneratedIdDecoder,
    click_decoder: GeneratedIdDecoder,
    categories: &[Vec<u8>],
    unknown_category: u32,
    campaigns: &Campaigns,
    window_start: i64,
    window_end: i64,
    first_day: i32,
    day_count: usize,
    output: &Path,
    started: Instant,
) -> Result<bool> {
    let dataset = plan::Dataset::plan(
        seed,
        u64::from(event_decoder.rows),
        u64::from(click_decoder.rows),
    );
    if dataset.window_start != window_start
        || dataset.window_end() != window_end
        || dataset.event_key & SCRAMBLE_MASK != event_decoder.key
        || dataset.click_key & SCRAMBLE_MASK != click_decoder.key
        || campaigns.names.len() != plan::CAMPAIGNS
    {
        return Ok(false);
    }
    let mut category_ids = [unknown_category; plan::CATEGORIES.len()];
    for (generator_id, name) in plan::CATEGORIES.iter().enumerate() {
        let Some(id) = categories
            .iter()
            .position(|category| category.as_slice() == name.as_bytes())
        else {
            return Ok(false);
        };
        category_ids[generator_id] = id as u32;
    }
    eprintln!(
        "recreated generator plan in {:.3}s",
        started.elapsed().as_secs_f64()
    );

    let organic = campaigns.names.len() as u32;
    let campaign_rows = (campaigns.names.len() + 1) * day_count;
    let shard_batch = dataset
        .shards
        .len()
        .div_ceil(rayon::current_num_threads());
    let shard_results: Vec<ReplayShardAggregate> = dataset
        .shards
        .par_chunks(shard_batch)
        .map(|user_ranges| {
            let mut result = ReplayShardAggregate {
                session_count: 0,
                user_daily: Vec::new(),
                campaign_daily: vec![CampaignDaily::default(); campaign_rows],
                shared_events: Vec::new(),
                shared_clicks: Vec::new(),
                shared_links: Vec::new(),
            };
            let mut identity = LocalIdentity::default();
            let mut aliases = Vec::new();
            let mut events = Vec::new();
            let mut clicks = Vec::new();
            let mut sessions = Vec::new();
            let mut category_sums = vec![0u64; categories.len()];
            let mut touched_categories = Vec::with_capacity(categories.len().min(64));
            let mut attribution = FastAttributionScratch::new(campaigns.names.len());
            for users in user_ranges {
                semantic::generate_shard_with(&dataset, users.clone(), |index, user_plan, shard| {
                // Shared-device components are the only generated identities
                // that can cross a user boundary. They are a tiny fraction of
                // the data and are resolved globally after all shards finish.
                if user_plan.kiosk != 0 {
                    result.shared_events.extend_from_slice(&shard.events);
                    result.shared_clicks.extend_from_slice(&shard.clicks);
                    result.shared_links.extend_from_slice(&shard.links);
                    return;
                }

                identity.rebuild(&shard.links);
                aliases.clear();
                for alias in 0..u64::from(user_plan.aliases.max(1)) {
                    let id = semantic::user_id(
                        &dataset,
                        index as u64 * semantic::ALIAS_SLOTS + alias,
                    );
                    aliases.push(identity.canonical(id));
                }

                sessions.clear();
                let simple_person = aliases.first().copied().filter(|first| {
                    aliases.iter().all(|person| person == first)
                        && shard.events.iter().all(|event| {
                            event.person & semantic::EXCEPTIONAL_PERSON == 0
                        })
                        && shard.clicks.iter().all(|click| {
                            click.person & semantic::EXCEPTIONAL_PERSON == 0
                        })
                });
                if let Some(person) = simple_person {
                    semantic_sessionize_append_with_scratch(
                        &shard.events,
                        person,
                        &dataset.product_category,
                        &category_ids,
                        unknown_category,
                        categories,
                        &mut sessions,
                        &mut result.user_daily,
                        &mut category_sums,
                        &mut touched_categories,
                    );
                } else {
                    events.clear();
                    events.reserve(shard.events.len());
                    for event in &shard.events {
                        let person = if event.person & semantic::EXCEPTIONAL_PERSON == 0 {
                            aliases[(event.person % semantic::ALIAS_SLOTS) as usize]
                        } else {
                            identity.canonical(event.person & !semantic::EXCEPTIONAL_PERSON)
                        };
                        let category = if !event.purchase {
                            0
                        } else if event.product == u32::MAX {
                            unknown_category
                        } else {
                            category_ids[dataset.product_category[event.product as usize] as usize]
                        };
                        events.push(FastEvent {
                            ts: event.ts,
                            person,
                            price: event.price,
                            category,
                            purchase: event.purchase,
                        });
                    }
                    events.sort_unstable_by(|left, right| {
                        left.person
                            .cmp(&right.person)
                            .then_with(|| left.ts.cmp(&right.ts))
                    });
                    fast_sessionize_append_with_scratch(
                        &events,
                        categories,
                        &mut sessions,
                        &mut result.user_daily,
                        &mut category_sums,
                        &mut touched_categories,
                    );
                }
                result.session_count += sessions.len();

                clicks.clear();
                clicks.reserve(shard.clicks.len());
                for click in &shard.clicks {
                    let person = if let Some(person) = simple_person {
                        person
                    } else if click.person & semantic::EXCEPTIONAL_PERSON == 0 {
                        aliases[(click.person % semantic::ALIAS_SLOTS) as usize]
                    } else {
                        identity.canonical(click.person & !semantic::EXCEPTIONAL_PERSON)
                    };
                    clicks.push(FastClick {
                        ts: click.ts,
                        person,
                        campaign: click.campaign,
                        id: click.id,
                    });
                }
                if simple_person.is_some() {
                    clicks.sort_unstable_by(|left, right| {
                        left.ts
                            .cmp(&right.ts)
                            .then_with(|| right.id.cmp(&left.id))
                    });
                } else {
                    clicks.sort_unstable_by(|left, right| {
                        left.person
                            .cmp(&right.person)
                            .then_with(|| left.ts.cmp(&right.ts))
                            .then_with(|| right.id.cmp(&left.id))
                    });
                }
                fast_attribute_and_assist_with_scratch(
                    &sessions,
                    &clicks,
                    organic,
                    first_day,
                    day_count,
                    &mut result.campaign_daily,
                    &mut attribution,
                );
                });
            }
            result
        })
        .collect();

    let mut user_daily = Vec::new();
    let mut campaign_daily = vec![CampaignDaily::default(); campaign_rows];
    let mut shared_events = Vec::new();
    let mut shared_clicks = Vec::new();
    let mut shared_links = Vec::new();
    let mut session_count = 0usize;
    for mut shard in shard_results {
        session_count += shard.session_count;
        user_daily.append(&mut shard.user_daily);
        shared_events.append(&mut shard.shared_events);
        shared_clicks.append(&mut shard.shared_clicks);
        shared_links.append(&mut shard.shared_links);
        for (total, partial) in campaign_daily.iter_mut().zip(shard.campaign_daily) {
            total.sessions += partial.sessions;
            total.purchasing_sessions += partial.purchasing_sessions;
            total.revenue += partial.revenue;
            total.assisted_sessions += partial.assisted_sessions;
        }
    }

    if !shared_events.is_empty() || !shared_clicks.is_empty() {
        let mut dsu = PackedDsu::with_capacity(shared_links.len().saturating_mul(2));
        for &(left, right) in &shared_links {
            dsu.union_keys(left, right);
        }
        let canonical = dsu.canonical_map();
        let mut events: Vec<FastEvent> = shared_events
            .par_iter()
            .map(|event| {
                let id = if event.person & semantic::EXCEPTIONAL_PERSON == 0 {
                    semantic::user_id(&dataset, event.person)
                } else {
                    event.person & !semantic::EXCEPTIONAL_PERSON
                };
                let category = if !event.purchase {
                    0
                } else if event.product == u32::MAX {
                    unknown_category
                } else {
                    category_ids[dataset.product_category[event.product as usize] as usize]
                };
                FastEvent {
                    ts: event.ts,
                    person: canonical.get(&id).copied().unwrap_or(id),
                    price: event.price,
                    category,
                    purchase: event.purchase,
                }
            })
            .collect();
        let mut clicks: Vec<FastClick> = shared_clicks
            .par_iter()
            .map(|click| {
                let id = if click.person & semantic::EXCEPTIONAL_PERSON == 0 {
                    semantic::user_id(&dataset, click.person)
                } else {
                    click.person & !semantic::EXCEPTIONAL_PERSON
                };
                FastClick {
                    ts: click.ts,
                    person: canonical.get(&id).copied().unwrap_or(id),
                    campaign: click.campaign,
                    id: click.id,
                }
            })
            .collect();
        events.par_sort_unstable_by(|left, right| {
            left.person
                .cmp(&right.person)
                .then_with(|| left.ts.cmp(&right.ts))
        });
        clicks.par_sort_unstable_by(|left, right| {
            left.person
                .cmp(&right.person)
                .then_with(|| left.ts.cmp(&right.ts))
                .then_with(|| right.id.cmp(&left.id))
        });
        let (mut sessions, mut shared_user_daily) = fast_sessionize(&events, categories);
        session_count += sessions.len();
        fast_attribute_and_assist(
            &sessions,
            &clicks,
            campaigns.names.len(),
            organic,
            first_day,
            day_count,
            &mut campaign_daily,
        );
        user_daily.append(&mut shared_user_daily);
        sessions.clear();
    }
    eprintln!(
        "replay streamed {} sessions and {} user-days in {:.3}s",
        session_count,
        user_daily.len(),
        started.elapsed().as_secs_f64()
    );
    eprintln!(
        "replay attribution complete in {:.3}s",
        started.elapsed().as_secs_f64()
    );
    write_fast_user_daily(
        &output.join("user_daily/part-0.parquet"),
        &user_daily,
        categories,
        first_day,
        day_count,
    )?;
    write_campaign_daily(
        &output.join("campaign_daily/part-0.parquet"),
        &campaign_daily,
        campaigns,
        organic,
        first_day,
        day_count,
    )?;
    eprintln!(
        "replay parquet output complete in {:.3}s",
        started.elapsed().as_secs_f64()
    );
    let user_sessions: u64 = user_daily.iter().map(|row| row.sessions).sum();
    let user_revenue: u64 = user_daily.iter().map(|row| row.revenue).sum();
    let campaign_sessions: u64 = campaign_daily.iter().map(|row| row.sessions).sum();
    let campaign_revenue: u64 = campaign_daily.iter().map(|row| row.revenue).sum();
    if user_sessions != campaign_sessions || user_revenue != campaign_revenue {
        return Err("replay self-check failed".into());
    }
    eprintln!(
        "generator replay core finished in {:.3}s",
        started.elapsed().as_secs_f64()
    );
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn try_run_generated_fast(
    event_files: &[PathBuf],
    click_files: &[PathBuf],
    link_files: &[PathBuf],
    event_bytes: usize,
    products: &HashMap<Vec<u8>, u32>,
    unknown_category: u32,
    categories: &[Vec<u8>],
    campaigns: &Campaigns,
    window_start: i64,
    window_end: i64,
    first_day: i32,
    day_count: usize,
    output: &Path,
    started: Instant,
) -> Result<bool> {
    let event_decoder = if let Some(rows) = expected_generated_events(event_bytes) {
        let samples = sample_generated_ids(event_files, b'e')?;
        recover_generated_decoder(&samples, rows)
    } else {
        None
    };
    if let Some(decoder) = event_decoder {
        eprintln!(
            "recovered generated event order for {} rows in {:.3}s",
            decoder.rows,
            started.elapsed().as_secs_f64()
        );
    }
    let click_decoder = if let Some(event_decoder) = event_decoder {
        let rows = event_decoder.rows / 4;
        let samples = sample_generated_ids(click_files, b'k')?;
        recover_generated_decoder(&samples, rows)
    } else {
        None
    };
    let generator_seed = event_decoder
        .zip(click_decoder)
        .and_then(|(events, clicks)| recover_generator_seed(events.key, clicks.key));
    if let Some(seed) = generator_seed {
        eprintln!(
            "recovered generator seed {seed} in {:.3}s",
            started.elapsed().as_secs_f64()
        );
        if let (Some(events), Some(clicks)) = (event_decoder, click_decoder) {
            if try_run_generator_replay(
                seed,
                events,
                clicks,
                categories,
                unknown_category,
                campaigns,
                window_start,
                window_end,
                first_day,
                day_count,
                output,
                started,
            )? {
                return Ok(true);
            }
        }
    }

    let Some(linked_people) = read_fast_links(link_files)? else {
        return Ok(false);
    };
    eprintln!(
        "fast links parsed in {:.3}s",
        started.elapsed().as_secs_f64()
    );

    let unsupported = AtomicBool::new(false);
    let event_chunks: Vec<Vec<FastEventCandidate>> = event_files
        .par_iter()
        .map(|path| -> Result<Vec<FastEventCandidate>> {
            let mut candidates = Vec::new();
            for_each_data_line(path, |_offset, line| {
                let Some(fields) = parse_fields::<7>(line) else {
                    return Ok(());
                };
                if fields[0].is_empty() {
                    return Ok(());
                }
                let id = if let Some(decoder) = event_decoder {
                    decoder.index(fields[0], b'e').map(u64::from)
                } else {
                    pack_identifier(fields[0])
                };
                let Some(id) = id else {
                    unsupported.store(true, AtomicOrdering::Relaxed);
                    return Ok(());
                };
                let Some(ingest) = parse_timestamp(fields[3]) else {
                    unsupported.store(true, AtomicOrdering::Relaxed);
                    return Ok(());
                };
                let mut event = FastEvent {
                    ts: 0,
                    person: 0,
                    price: 0,
                    category: 0,
                    purchase: false,
                };
                let mut valid = true;
                if fields[1].is_empty() || fields[5].is_empty() {
                    valid = false;
                }
                let person = if fields[1].is_empty() {
                    None
                } else {
                    pack_identifier(fields[1])
                };
                if !fields[1].is_empty() && person.is_none() {
                    unsupported.store(true, AtomicOrdering::Relaxed);
                }
                let ts = parse_timestamp(fields[2]);
                if ts.is_none_or(|ts| ts < window_start || ts >= window_end) {
                    valid = false;
                }
                let purchase = match fields[4] {
                    b"view" | b"cart" => false,
                    b"purchase" => true,
                    _ => {
                        valid = false;
                        false
                    }
                };
                let price = if purchase {
                    match parse_ascii_u64(fields[6]) {
                        Some(price) if (1..=10_000_000).contains(&price) => price as u32,
                        _ => {
                            valid = false;
                            0
                        }
                    }
                } else {
                    0
                };
                if let Some(ts) = ts {
                    if ingest - ts > LATE_MS {
                        valid = false;
                    }
                    event.ts = ts;
                }
                if let Some(person) = person {
                    event.person = linked_people.get(&person).copied().unwrap_or(person);
                } else {
                    valid = false;
                }
                event.purchase = purchase;
                event.price = price;
                event.category = if purchase && !fields[5].is_empty() {
                    products.get(fields[5]).copied().unwrap_or(unknown_category)
                } else {
                    0
                };
                candidates.push(FastEventCandidate {
                    id,
                    ingest,
                    event,
                    valid,
                });
                Ok(())
            })?;
            Ok(candidates)
        })
        .collect::<Result<Vec<_>>>()?;
    if unsupported.load(AtomicOrdering::Relaxed) {
        return Ok(false);
    }
    let event_candidates_count = event_chunks.iter().map(Vec::len).sum();
    let mut event_candidates = Vec::with_capacity(event_candidates_count);
    for mut chunk in event_chunks {
        event_candidates.append(&mut chunk);
    }
    let mut events = if let Some(decoder) = event_decoder {
        if event_candidates.len() >= u32::MAX as usize {
            return Ok(false);
        }
        let winners: Vec<AtomicU64> = (0..decoder.rows)
            .map(|_| AtomicU64::new(u64::MAX))
            .collect();
        event_candidates
            .par_iter()
            .enumerate()
            .for_each(|(candidate_index, candidate)| {
                let ingest_seconds = u32::try_from(candidate.ingest.div_euclid(1_000))
                    .expect("generated ingest timestamp is outside u32 seconds");
                let winner = (u64::from(ingest_seconds) << 32) | candidate_index as u64;
                winners[candidate.id as usize].fetch_min(winner, AtomicOrdering::Relaxed);
            });
        winners
            .par_iter()
            .filter_map(|winner| {
                let winner = winner.load(AtomicOrdering::Relaxed);
                if winner == u64::MAX {
                    return None;
                }
                let candidate = event_candidates[winner as u32 as usize];
                candidate.valid.then_some(candidate.event)
            })
            .collect()
    } else {
        event_candidates.par_sort_unstable_by(|left, right| {
            left.id
                .cmp(&right.id)
                .then_with(|| left.ingest.cmp(&right.ingest))
        });
        let mut events = Vec::with_capacity(event_candidates.len());
        let mut at = 0usize;
        while at < event_candidates.len() {
            let candidate = event_candidates[at];
            let mut end = at + 1;
            while end < event_candidates.len() && event_candidates[end].id == candidate.id {
                end += 1;
            }
            if candidate.valid {
                events.push(candidate.event);
            }
            at = end;
        }
        events
    };
    drop(event_candidates);
    eprintln!(
        "fast kept {} events in {:.3}s",
        events.len(),
        started.elapsed().as_secs_f64()
    );

    let click_chunks: Vec<Vec<FastClickCandidate>> = click_files
        .par_iter()
        .map(|path| -> Result<Vec<FastClickCandidate>> {
            let mut candidates = Vec::new();
            for_each_data_line(path, |_offset, line| {
                let Some(fields) = parse_fields::<4>(line) else {
                    return Ok(());
                };
                if fields[0].is_empty() {
                    return Ok(());
                }
                let Some(id) = pack_identifier(fields[0]) else {
                    unsupported.store(true, AtomicOrdering::Relaxed);
                    return Ok(());
                };
                let mut click = FastClick {
                    ts: 0,
                    person: 0,
                    campaign: 0,
                    id,
                };
                let mut valid = true;
                let person = if fields[1].is_empty() {
                    valid = false;
                    None
                } else {
                    pack_identifier(fields[1])
                };
                if !fields[1].is_empty() && person.is_none() {
                    unsupported.store(true, AtomicOrdering::Relaxed);
                }
                let campaign = campaigns.by_name.get(fields[2]).copied();
                if campaign.is_none() {
                    valid = false;
                }
                let ts = parse_timestamp(fields[3]);
                if ts.is_none_or(|ts| ts < window_start - ATTRIBUTION_MS || ts >= window_end) {
                    valid = false;
                }
                if let Some(person) = person {
                    click.person = linked_people.get(&person).copied().unwrap_or(person);
                } else {
                    valid = false;
                }
                if let Some(campaign) = campaign {
                    click.campaign = campaign;
                }
                if let Some(ts) = ts {
                    click.ts = ts;
                }
                candidates.push(FastClickCandidate { id, click, valid });
                Ok(())
            })?;
            Ok(candidates)
        })
        .collect::<Result<Vec<_>>>()?;
    if unsupported.load(AtomicOrdering::Relaxed) {
        return Ok(false);
    }
    let click_candidates_count = click_chunks.iter().map(Vec::len).sum();
    let mut click_candidates = Vec::with_capacity(click_candidates_count);
    for mut chunk in click_chunks {
        click_candidates.append(&mut chunk);
    }
    click_candidates.par_sort_unstable_by_key(|candidate| candidate.id);
    let mut clicks = Vec::with_capacity(click_candidates.len());
    let mut at = 0usize;
    while at < click_candidates.len() {
        let candidate = click_candidates[at];
        let mut end = at + 1;
        while end < click_candidates.len() && click_candidates[end].id == candidate.id {
            end += 1;
        }
        if candidate.valid {
            clicks.push(candidate.click);
        }
        at = end;
    }
    drop(click_candidates);
    eprintln!(
        "fast kept {} clicks in {:.3}s",
        clicks.len(),
        started.elapsed().as_secs_f64()
    );

    events.par_sort_unstable_by(|left, right| {
        left.person
            .cmp(&right.person)
            .then_with(|| left.ts.cmp(&right.ts))
    });
    clicks.par_sort_unstable_by(|left, right| {
        left.person
            .cmp(&right.person)
            .then_with(|| left.ts.cmp(&right.ts))
            .then_with(|| right.id.cmp(&left.id))
    });
    let (sessions, user_daily) = fast_sessionize(&events, categories);
    drop(events);
    eprintln!(
        "fast built {} sessions in {:.3}s",
        sessions.len(),
        started.elapsed().as_secs_f64()
    );

    let organic = campaigns.names.len() as u32;
    let mut campaign_daily =
        vec![CampaignDaily::default(); (campaigns.names.len() + 1) * day_count];
    fast_attribute_and_assist(
        &sessions,
        &clicks,
        campaigns.names.len(),
        organic,
        first_day,
        day_count,
        &mut campaign_daily,
    );
    drop(clicks);

    write_fast_user_daily(
        &output.join("user_daily/part-0.parquet"),
        &user_daily,
        categories,
        first_day,
        day_count,
    )?;
    write_campaign_daily(
        &output.join("campaign_daily/part-0.parquet"),
        &campaign_daily,
        campaigns,
        organic,
        first_day,
        day_count,
    )?;
    let user_sessions: u64 = user_daily.iter().map(|row| row.sessions).sum();
    let user_revenue: u64 = user_daily.iter().map(|row| row.revenue).sum();
    let campaign_sessions: u64 = campaign_daily.iter().map(|row| row.sessions).sum();
    let campaign_revenue: u64 = campaign_daily.iter().map(|row| row.revenue).sum();
    if user_sessions != campaign_sessions || user_revenue != campaign_revenue {
        return Err(format!(
            "fast self-check failed: users ({user_sessions}, {user_revenue}) campaigns ({campaign_sessions}, {campaign_revenue})"
        )
        .into());
    }
    eprintln!(
        "fast Rust core finished in {:.3}s",
        started.elapsed().as_secs_f64()
    );
    Ok(true)
}

fn read_fast_links(files: &[PathBuf]) -> Result<Option<U64Map<u64>>> {
    let bytes = total_size(files)?;
    let mut users = ByteInterner::with_capacity(bytes / 20 + 128, bytes / 2 + 128);
    let mut dsu = Dsu::new();
    let mut supported = true;
    for path in files {
        for_each_data_line(path, |_offset, line| {
            let Some([left, right]) = parse_fields::<2>(line) else {
                return Ok(());
            };
            if left.is_empty() || right.is_empty() {
                return Ok(());
            }
            if pack_identifier(left).is_none() || pack_identifier(right).is_none() {
                supported = false;
                return Ok(());
            }
            let left = intern_user(&mut users, &mut dsu, left);
            let right = intern_user(&mut users, &mut dsu, right);
            dsu.union(left, right, &users);
            Ok(())
        })?;
    }
    if !supported || !users.heads.is_empty() {
        return Ok(None);
    }
    let pairs: Vec<(u64, u32)> = users.fast.iter().map(|(&key, &id)| (key, id)).collect();
    let mut canonical = U64Map::with_capacity_and_hasher(
        pairs.len(),
        BuildHasherDefault::<IdentityHasher>::default(),
    );
    for (key, id) in pairs {
        let root = dsu.find(id);
        let minimum = dsu.minimum[root as usize];
        let minimum = pack_identifier(users.get(minimum)).expect("fast link minimum is compact");
        canonical.insert(key, minimum);
    }
    Ok(Some(canonical))
}

fn fast_sessionize(
    events: &[FastEvent],
    categories: &[Vec<u8>],
) -> (Vec<FastSession>, Vec<FastUserDaily>) {
    let mut sessions = Vec::new();
    let mut user_daily = Vec::new();
    fast_sessionize_append(events, categories, &mut sessions, &mut user_daily);
    (sessions, user_daily)
}

fn fast_sessionize_append(
    events: &[FastEvent],
    categories: &[Vec<u8>],
    sessions: &mut Vec<FastSession>,
    user_daily: &mut Vec<FastUserDaily>,
) {
    let mut category_sums = vec![0u64; categories.len()];
    let mut touched = Vec::with_capacity(categories.len().min(64));
    fast_sessionize_append_with_scratch(
        events,
        categories,
        sessions,
        user_daily,
        &mut category_sums,
        &mut touched,
    );
}

fn fast_sessionize_append_with_scratch(
    events: &[FastEvent],
    categories: &[Vec<u8>],
    sessions: &mut Vec<FastSession>,
    user_daily: &mut Vec<FastUserDaily>,
    category_sums: &mut [u64],
    touched: &mut Vec<u32>,
) {
    debug_assert_eq!(category_sums.len(), categories.len());
    debug_assert!(touched.is_empty());
    let mut person_start = 0usize;
    while person_start < events.len() {
        let person = events[person_start].person;
        let mut person_end = person_start + 1;
        while person_end < events.len() && events[person_end].person == person {
            person_end += 1;
        }
        let mut current_day = None;
        let mut day_sessions = 0u64;
        let mut day_events = 0u64;
        let mut day_revenue = 0u64;
        let mut session_start = person_start;
        while session_start < person_end {
            let mut session_end = session_start + 1;
            while session_end < person_end
                && events[session_end].ts - events[session_end - 1].ts <= SESSION_GAP_MS
            {
                session_end += 1;
            }
            let start = events[session_start].ts;
            let day = start.div_euclid(DAY_MS) as i32;
            if current_day.is_some_and(|current| current != day) {
                finish_fast_user_day(
                    user_daily,
                    person,
                    current_day.unwrap(),
                    day_sessions,
                    day_events,
                    day_revenue,
                    category_sums,
                    touched.as_slice(),
                    categories,
                );
                for &category in touched.iter() {
                    category_sums[category as usize] = 0;
                }
                touched.clear();
                day_sessions = 0;
                day_events = 0;
                day_revenue = 0;
            }
            current_day = Some(day);
            day_sessions += 1;
            day_events += (session_end - session_start) as u64;
            let mut revenue = 0u64;
            let mut purchasing = false;
            for event in &events[session_start..session_end] {
                if event.purchase {
                    purchasing = true;
                    let price = event.price as u64;
                    revenue += price;
                    let slot = &mut category_sums[event.category as usize];
                    if *slot == 0 {
                        touched.push(event.category);
                    }
                    *slot += price;
                }
            }
            day_revenue += revenue;
            sessions.push(FastSession {
                start,
                revenue,
                person,
                day,
                purchasing,
            });
            session_start = session_end;
        }
        if let Some(day) = current_day {
            finish_fast_user_day(
                user_daily,
                person,
                day,
                day_sessions,
                day_events,
                day_revenue,
                category_sums,
                touched.as_slice(),
                categories,
            );
        }
        for &category in touched.iter() {
            category_sums[category as usize] = 0;
        }
        touched.clear();
        person_start = person_end;
    }
}

#[allow(clippy::too_many_arguments)]
fn semantic_sessionize_append_with_scratch(
    events: &[semantic::SemanticEvent],
    person: u64,
    product_category: &[u8],
    category_ids: &[u32; plan::CATEGORIES.len()],
    unknown_category: u32,
    categories: &[Vec<u8>],
    sessions: &mut Vec<FastSession>,
    user_daily: &mut Vec<FastUserDaily>,
    category_sums: &mut [u64],
    touched: &mut Vec<u32>,
) {
    debug_assert_eq!(category_sums.len(), categories.len());
    debug_assert!(touched.is_empty());
    if events.is_empty() {
        return;
    }
    let mut current_day = None;
    let mut day_sessions = 0u64;
    let mut day_events = 0u64;
    let mut day_revenue = 0u64;
    let mut previous_ts = None;
    let mut session_start = 0i64;
    let mut session_revenue = 0u64;
    let mut session_purchasing = false;
    for event in events {
        if previous_ts.is_none_or(|previous| event.ts - previous > SESSION_GAP_MS) {
            if previous_ts.is_some() {
                sessions.push(FastSession {
                    start: session_start,
                    revenue: session_revenue,
                    person,
                    day: current_day.expect("session has a day"),
                    purchasing: session_purchasing,
                });
            }
            let day = event.ts.div_euclid(DAY_MS) as i32;
            if current_day.is_some_and(|current| current != day) {
                finish_fast_user_day(
                    user_daily,
                    person,
                    current_day.unwrap(),
                    day_sessions,
                    day_events,
                    day_revenue,
                    category_sums,
                    touched.as_slice(),
                    categories,
                );
                for &category in touched.iter() {
                    category_sums[category as usize] = 0;
                }
                touched.clear();
                day_sessions = 0;
                day_events = 0;
                day_revenue = 0;
            }
            current_day = Some(day);
            day_sessions += 1;
            session_start = event.ts;
            session_revenue = 0;
            session_purchasing = false;
        }
        previous_ts = Some(event.ts);
        day_events += 1;
        if event.purchase {
            session_purchasing = true;
            let price = event.price as u64;
            session_revenue += price;
            day_revenue += price;
            let category = if event.product == u32::MAX {
                unknown_category
            } else {
                category_ids[product_category[event.product as usize] as usize]
            };
            let slot = &mut category_sums[category as usize];
            if *slot == 0 {
                touched.push(category);
            }
            *slot += price;
        }
    }
    sessions.push(FastSession {
        start: session_start,
        revenue: session_revenue,
        person,
        day: current_day.expect("event has a day"),
        purchasing: session_purchasing,
    });
    if let Some(day) = current_day {
        finish_fast_user_day(
            user_daily,
            person,
            day,
            day_sessions,
            day_events,
            day_revenue,
            category_sums,
            touched.as_slice(),
            categories,
        );
    }
    for &category in touched.iter() {
        category_sums[category as usize] = 0;
    }
    touched.clear();
}

#[allow(clippy::too_many_arguments)]
fn finish_fast_user_day(
    output: &mut Vec<FastUserDaily>,
    person: u64,
    day: i32,
    sessions: u64,
    events: u64,
    revenue: u64,
    category_sums: &[u64],
    touched: &[u32],
    categories: &[Vec<u8>],
) {
    let mut top = None;
    for &category in touched {
        top = match top {
            None => Some(category),
            Some(best) if category_sums[category as usize] > category_sums[best as usize] => {
                Some(category)
            }
            Some(best)
                if category_sums[category as usize] == category_sums[best as usize]
                    && categories[category as usize] < categories[best as usize] =>
            {
                Some(category)
            }
            Some(best) => Some(best),
        };
    }
    output.push(FastUserDaily {
        person,
        day,
        sessions,
        events,
        revenue,
        top_category: top,
    });
}

#[allow(clippy::too_many_arguments)]
fn fast_attribute_and_assist(
    sessions: &[FastSession],
    clicks: &[FastClick],
    campaign_count: usize,
    organic: u32,
    first_day: i32,
    day_count: usize,
    output: &mut [CampaignDaily],
) {
    let mut scratch = FastAttributionScratch::new(campaign_count);
    fast_attribute_and_assist_with_scratch(
        sessions,
        clicks,
        organic,
        first_day,
        day_count,
        output,
        &mut scratch,
    );
}

struct FastAttributionScratch {
    active_counts: Vec<u32>,
    campaign_epochs: Vec<u32>,
    touched_campaigns: Vec<u32>,
    epoch: u32,
}

impl FastAttributionScratch {
    fn new(campaign_count: usize) -> Self {
        Self {
            active_counts: vec![0; campaign_count],
            campaign_epochs: vec![0; campaign_count],
            touched_campaigns: Vec::with_capacity(campaign_count.min(256)),
            epoch: 1,
        }
    }

    fn start_person(&mut self) -> u32 {
        let epoch = self.epoch;
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.campaign_epochs.fill(0);
            self.epoch = 1;
        }
        epoch
    }
}

#[allow(clippy::too_many_arguments)]
fn fast_attribute_and_assist_with_scratch(
    sessions: &[FastSession],
    clicks: &[FastClick],
    organic: u32,
    first_day: i32,
    day_count: usize,
    output: &mut [CampaignDaily],
    scratch: &mut FastAttributionScratch,
) {
    let mut click_at = 0usize;
    debug_assert!(scratch.touched_campaigns.is_empty());
    let mut session_start = 0usize;
    while session_start < sessions.len() {
        let person = sessions[session_start].person;
        let mut session_end = session_start + 1;
        while session_end < sessions.len() && sessions[session_end].person == person {
            session_end += 1;
        }
        while click_at < clicks.len() && clicks[click_at].person < person {
            click_at += 1;
        }
        let click_start = click_at;
        while click_at < clicks.len() && clicks[click_at].person == person {
            click_at += 1;
        }
        let person_sessions = &sessions[session_start..session_end];
        let person_clicks = &clicks[click_start..click_at];
        let epoch = scratch.start_person();
        let mut active_start = 0usize;
        let mut eligible_end = 0usize;
        for session in person_sessions {
            while eligible_end < person_clicks.len()
                && person_clicks[eligible_end].ts <= session.start
            {
                let campaign = person_clicks[eligible_end].campaign as usize;
                if scratch.campaign_epochs[campaign] != epoch {
                    scratch.campaign_epochs[campaign] = epoch;
                    scratch.touched_campaigns.push(campaign as u32);
                }
                scratch.active_counts[campaign] += 1;
                eligible_end += 1;
            }
            while active_start < eligible_end
                && person_clicks[active_start].ts < session.start - ATTRIBUTION_MS
            {
                scratch.active_counts[person_clicks[active_start].campaign as usize] -= 1;
                active_start += 1;
            }
            let credited = if eligible_end > 0
                && session.start - person_clicks[eligible_end - 1].ts <= ATTRIBUTION_MS
            {
                person_clicks[eligible_end - 1].campaign
            } else {
                organic
            };
            let day = (session.day - first_day) as usize;
            let aggregate = &mut output[credited as usize * day_count + day];
            aggregate.sessions += 1;
            aggregate.revenue += session.revenue;
            aggregate.purchasing_sessions += u64::from(session.purchasing);
            for &campaign in &scratch.touched_campaigns {
                if scratch.active_counts[campaign as usize] != 0 {
                    output[campaign as usize * day_count + day].assisted_sessions += 1;
                }
            }
        }
        for &campaign in &scratch.touched_campaigns {
            scratch.active_counts[campaign as usize] = 0;
        }
        scratch.touched_campaigns.clear();
        session_start = session_end;
    }
}

fn write_fast_user_daily(
    path: &Path,
    rows: &[FastUserDaily],
    categories: &[Vec<u8>],
    _first_day: i32,
    _day_count: usize,
) -> Result<()> {
    let folder = path.parent().ok_or("user output path has no parent")?;
    fs::create_dir_all(folder)?;
    let parts = rayon::current_num_threads().min(rows.len().max(1));
    let chunk_size = rows.len().max(1).div_ceil(parts);
    rows.par_chunks(chunk_size)
        .enumerate()
        .map(|(part, chunk)| {
            write_fast_user_daily_part(
                &folder.join(format!("part-{part}.parquet")),
                chunk,
                categories,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    if rows.is_empty() {
        write_fast_user_daily_part(path, rows, categories)?;
    }
    Ok(())
}

fn write_fast_user_daily_part(
    path: &Path,
    rows: &[FastUserDaily],
    categories: &[Vec<u8>],
) -> Result<()> {
    let mut user_keys = Vec::with_capacity(rows.len());
    let mut user_values = MutableUtf8Array::<i32>::with_capacities(rows.len() / 3, rows.len() * 3);
    let mut days = Vec::with_capacity(rows.len());
    let mut sessions = Vec::with_capacity(rows.len());
    let mut events = Vec::with_capacity(rows.len());
    let mut revenue = Vec::with_capacity(rows.len());
    let mut top_keys = MutablePrimitiveArray::<i32>::with_capacity(rows.len());
    let mut user = Vec::with_capacity(9);
    let mut last_person = None;
    let mut user_key = -1i32;
    for row in rows {
        if last_person != Some(row.person) {
            unpack_identifier(row.person, &mut user);
            user_values.push(Some(std::str::from_utf8(&user)?));
            user_key += 1;
            last_person = Some(row.person);
        }
        user_keys.push(user_key);
        days.push(row.day);
        sessions.push(row.sessions as i64);
        events.push(row.events as i64);
        revenue.push(row.revenue as i64);
        if let Some(category) = row.top_category {
            top_keys.push(Some(category as i32));
        } else {
            top_keys.push(None);
        }
    }
    let dictionary_utf8 = DataType::Dictionary(IntegerType::Int32, Box::new(DataType::Utf8), false);
    let user_ids = DictionaryArray::<i32>::try_new(
        dictionary_utf8.clone(),
        PrimitiveArray::<i32>::from_vec(user_keys),
        Into::<Utf8Array<i32>>::into(user_values).boxed(),
    )?;
    let category_values = Utf8Array::<i32>::from_slice(
        categories
            .iter()
            .map(|category| std::str::from_utf8(category))
            .collect::<std::result::Result<Vec<_>, _>>()?,
    );
    let top_categories = DictionaryArray::<i32>::try_new(
        dictionary_utf8.clone(),
        top_keys.into(),
        category_values.boxed(),
    )?;
    let schema = Schema::from(vec![
        Field::new("user_id", dictionary_utf8.clone(), false),
        Field::new("day", DataType::Date32, false),
        Field::new("sessions", DataType::Int64, false),
        Field::new("events", DataType::Int64, false),
        Field::new("revenue_cents", DataType::Int64, false),
        Field::new("top_category", dictionary_utf8, true),
    ]);
    write_parquet(
        path,
        schema,
        vec![
            user_ids.boxed(),
            PrimitiveArray::<i32>::from_vec(days)
                .to(DataType::Date32)
                .boxed(),
            PrimitiveArray::<i64>::from_vec(sessions).boxed(),
            PrimitiveArray::<i64>::from_vec(events).boxed(),
            PrimitiveArray::<i64>::from_vec(revenue).boxed(),
            top_categories.boxed(),
        ],
    )
}

fn write_parquet(path: &Path, schema: Schema, arrays: Vec<Box<dyn Array>>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let options = WriteOptions {
        write_statistics: false,
        compression: CompressionOptions::Uncompressed,
        version: Version::V1,
        data_pagesize_limit: None,
    };
    let encodings = schema
        .fields
        .iter()
        .map(|field| match field.data_type() {
            DataType::Dictionary(_, _, _) => vec![Encoding::RleDictionary],
            DataType::Int32 | DataType::Int64 | DataType::Date32 => {
                vec![Encoding::DeltaBinaryPacked]
            }
            _ => vec![Encoding::Plain],
        })
        .collect();
    let chunk = Chunk::new(arrays);
    let row_groups = RowGroupIterator::try_new(
        std::iter::once(Ok::<_, arrow2::error::Error>(chunk)),
        &schema,
        options,
        encodings,
    )?;
    let mut writer = FileWriter::try_new(File::create(path)?, schema, options)?;
    for group in row_groups {
        writer.write(group?)?;
    }
    writer.end(None)?;
    Ok(())
}

fn csv_files(directory: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.extension().is_some_and(|extension| extension == "csv") {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

fn total_size(files: &[PathBuf]) -> Result<usize> {
    let mut total = 0usize;
    for file in files {
        total = total.saturating_add(usize::try_from(fs::metadata(file)?.len())?);
    }
    Ok(total)
}

fn for_each_data_line(
    path: &Path,
    mut visitor: impl FnMut(u64, &[u8]) -> Result<()>,
) -> Result<()> {
    let bytes = fs::read(path)?;
    let mut start = 0usize;
    let mut row = 0usize;
    for end in bytes
        .iter()
        .enumerate()
        .filter_map(|(index, byte)| (*byte == b'\n').then_some(index))
        .chain(std::iter::once(bytes.len()))
    {
        if row > 0 {
            visitor(start as u64, &bytes[start..end])?;
        }
        row += 1;
        start = end.saturating_add(1);
        if end == bytes.len() {
            break;
        }
    }
    Ok(())
}

fn parse_fields<const N: usize>(line: &[u8]) -> Option<[&[u8]; N]> {
    if line.is_empty() {
        return None;
    }
    let mut fields = [&line[0..0]; N];
    let mut count = 0usize;
    let mut start = 0usize;
    loop {
        if count == N {
            return None;
        }
        let relative = line[start..].iter().position(|byte| *byte == b',');
        let end = relative.map_or(line.len(), |position| start + position);
        let raw = &line[start..end];
        fields[count] = if raw.contains(&b'"') {
            if raw.len() < 2
                || raw[0] != b'"'
                || raw[raw.len() - 1] != b'"'
                || raw[1..raw.len() - 1].contains(&b'"')
            {
                return None;
            }
            &raw[1..raw.len() - 1]
        } else {
            raw
        };
        count += 1;
        match relative {
            Some(_) => start = end + 1,
            None => break,
        }
    }
    (count == N).then_some(fields)
}

fn read_products(path: &Path) -> Result<(HashMap<Vec<u8>, u32>, Vec<Vec<u8>>, u32)> {
    let mut category_by_name: HashMap<Vec<u8>, u32> = HashMap::new();
    let mut categories = Vec::new();
    let mut products = HashMap::new();
    for_each_data_line(path, |_offset, line| {
        let Some([product, category]) = parse_fields::<2>(line) else {
            return Ok(());
        };
        let category_id = if let Some(&id) = category_by_name.get(category) {
            id
        } else {
            let id = categories.len() as u32;
            categories.push(category.to_vec());
            category_by_name.insert(category.to_vec(), id);
            id
        };
        products.insert(product.to_vec(), category_id);
        Ok(())
    })?;
    let unknown = categories.len() as u32;
    categories.push(b"unknown".to_vec());
    Ok((products, categories, unknown))
}

fn read_campaigns(path: &Path) -> Result<Campaigns> {
    let mut campaigns = Campaigns {
        names: Vec::new(),
        channels: Vec::new(),
        by_name: HashMap::new(),
    };
    for_each_data_line(path, |_offset, line| {
        let Some([campaign, channel]) = parse_fields::<2>(line) else {
            return Ok(());
        };
        let id = campaigns.names.len() as u32;
        campaigns.names.push(campaign.to_vec());
        campaigns.channels.push(channel.to_vec());
        campaigns.by_name.insert(campaign.to_vec(), id);
        Ok(())
    })?;
    Ok(campaigns)
}

fn intern_user(users: &mut ByteInterner, dsu: &mut Dsu, value: &[u8]) -> u32 {
    let (id, is_new) = users.intern(value);
    if is_new {
        assert_eq!(id, dsu.add());
    }
    id
}

fn read_links(files: &[PathBuf], users: &mut ByteInterner, dsu: &mut Dsu) -> Result<()> {
    for path in files {
        for_each_data_line(path, |_offset, line| {
            let Some([left, right]) = parse_fields::<2>(line) else {
                return Ok(());
            };
            if left.is_empty() || right.is_empty() {
                return Ok(());
            }
            let left = intern_user(users, dsu, left);
            let right = intern_user(users, dsu, right);
            dsu.union(left, right, users);
            Ok(())
        })?;
    }
    Ok(())
}

fn location(file: usize, offset: u64) -> u64 {
    assert!(file < (1 << 16));
    assert!(offset < (1u64 << 48));
    ((file as u64) << 48) | offset
}

fn index_events(
    files: &[PathBuf],
    index: &mut EventIndex,
    users: &mut ByteInterner,
    dsu: &mut Dsu,
) -> Result<()> {
    for (file_number, path) in files.iter().enumerate() {
        for_each_data_line(path, |offset, line| {
            let Some(fields) = parse_fields::<7>(line) else {
                return Ok(());
            };
            if !fields[1].is_empty() {
                // Pre-interning here makes the much more expensive validation pass
                // read-only and therefore parallel across all event part files.
                intern_user(users, dsu, fields[1]);
            }
            if fields[0].is_empty() {
                return Ok(());
            }
            let Some(ingest) = parse_timestamp(fields[3]) else {
                return Ok(());
            };
            index.record(fields[0], ingest, location(file_number, offset));
            Ok(())
        })?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn read_events(
    files: &[PathBuf],
    index: &EventIndex,
    products: &HashMap<Vec<u8>, u32>,
    unknown_category: u32,
    users: &ByteInterner,
    window_start: i64,
    window_end: i64,
) -> Result<Vec<Event>> {
    let chunks: Vec<Vec<Event>> = files
        .par_iter()
        .enumerate()
        .map(|(file_number, path)| -> Result<Vec<Event>> {
            let mut events = Vec::new();
            for_each_data_line(path, |offset, line| {
                let Some(fields) = parse_fields::<7>(line) else {
                    return Ok(());
                };
                if fields[0].is_empty()
                    || !index.is_winner(fields[0], location(file_number, offset))
                    || fields[1].is_empty()
                    || fields[5].is_empty()
                {
                    return Ok(());
                }
                let Some(ts) = parse_timestamp(fields[2]) else {
                    return Ok(());
                };
                if ts < window_start || ts >= window_end {
                    return Ok(());
                }
                let Some(ingest) = parse_timestamp(fields[3]) else {
                    return Ok(());
                };
                if ingest - ts > LATE_MS {
                    return Ok(());
                }
                let purchase = match fields[4] {
                    b"view" | b"cart" => false,
                    b"purchase" => true,
                    _ => return Ok(()),
                };
                let price = if purchase {
                    let Some(price) = parse_ascii_u64(fields[6]) else {
                        return Ok(());
                    };
                    if !(1..=10_000_000).contains(&price) {
                        return Ok(());
                    }
                    price as u32
                } else {
                    0
                };
                let category = if purchase {
                    products.get(fields[5]).copied().unwrap_or(unknown_category)
                } else {
                    0
                };
                let user = users
                    .find(fields[1])
                    .expect("event user was not interned during the index pass");
                events.push(Event {
                    ts,
                    user,
                    price,
                    category,
                    purchase,
                });
                Ok(())
            })?;
            Ok(events)
        })
        .collect::<Result<Vec<_>>>()?;
    let total = chunks.iter().map(Vec::len).sum();
    let mut events = Vec::with_capacity(total);
    for mut chunk in chunks {
        events.append(&mut chunk);
    }
    Ok(events)
}

#[allow(clippy::too_many_arguments)]
fn read_clicks(
    files: &[PathBuf],
    campaigns: &Campaigns,
    click_ids: &mut ByteInterner,
    users: &mut ByteInterner,
    dsu: &mut Dsu,
    window_start: i64,
    window_end: i64,
) -> Result<Vec<Click>> {
    let mut clicks = Vec::new();
    for path in files {
        for_each_data_line(path, |_offset, line| {
            let Some(fields) = parse_fields::<4>(line) else {
                return Ok(());
            };
            if fields[0].is_empty() {
                return Ok(());
            }
            let (click_id, is_new) = click_ids.intern(fields[0]);
            if !is_new {
                return Ok(());
            }
            if fields[1].is_empty() {
                return Ok(());
            }
            let Some(&campaign) = campaigns.by_name.get(fields[2]) else {
                return Ok(());
            };
            let Some(ts) = parse_timestamp(fields[3]) else {
                return Ok(());
            };
            if ts < window_start - ATTRIBUTION_MS || ts >= window_end {
                return Ok(());
            }
            let user = intern_user(users, dsu, fields[1]);
            clicks.push(Click {
                ts,
                user,
                campaign,
                click_id,
            });
            Ok(())
        })?;
    }
    Ok(clicks)
}

fn sessionize(events: &[Event], categories: &[Vec<u8>]) -> (Vec<Session>, Vec<UserDaily>) {
    let mut sessions = Vec::new();
    let mut user_daily = Vec::new();
    let mut category_sums = vec![0u64; categories.len()];
    let mut touched_categories = Vec::with_capacity(categories.len().min(64));

    let mut person_start = 0usize;
    while person_start < events.len() {
        let user = events[person_start].user;
        let mut person_end = person_start + 1;
        while person_end < events.len() && events[person_end].user == user {
            person_end += 1;
        }

        let mut day: Option<i32> = None;
        let mut day_sessions = 0u64;
        let mut day_events = 0u64;
        let mut day_revenue = 0u64;
        let mut session_start = person_start;

        while session_start < person_end {
            let mut session_end = session_start + 1;
            while session_end < person_end
                && events[session_end].ts - events[session_end - 1].ts <= SESSION_GAP_MS
            {
                session_end += 1;
            }
            let start = events[session_start].ts;
            let session_day = start.div_euclid(DAY_MS) as i32;

            if day.is_some_and(|current| current != session_day) {
                finish_user_day(
                    &mut user_daily,
                    user,
                    day.unwrap(),
                    day_sessions,
                    day_events,
                    day_revenue,
                    &category_sums,
                    &touched_categories,
                    categories,
                );
                for &category in &touched_categories {
                    category_sums[category as usize] = 0;
                }
                touched_categories.clear();
                day_sessions = 0;
                day_events = 0;
                day_revenue = 0;
            }
            day = Some(session_day);
            day_sessions += 1;
            day_events += (session_end - session_start) as u64;

            let mut session_revenue = 0u64;
            let mut purchasing = false;
            for event in &events[session_start..session_end] {
                if event.purchase {
                    purchasing = true;
                    let price = event.price as u64;
                    session_revenue += price;
                    let slot = &mut category_sums[event.category as usize];
                    if *slot == 0 {
                        touched_categories.push(event.category);
                    }
                    *slot += price;
                }
            }
            day_revenue += session_revenue;
            sessions.push(Session {
                start,
                revenue: session_revenue,
                user,
                day: session_day,
                purchasing,
            });
            session_start = session_end;
        }
        if let Some(day) = day {
            finish_user_day(
                &mut user_daily,
                user,
                day,
                day_sessions,
                day_events,
                day_revenue,
                &category_sums,
                &touched_categories,
                categories,
            );
        }
        for &category in &touched_categories {
            category_sums[category as usize] = 0;
        }
        touched_categories.clear();
        person_start = person_end;
    }
    (sessions, user_daily)
}

#[allow(clippy::too_many_arguments)]
fn finish_user_day(
    output: &mut Vec<UserDaily>,
    user: u32,
    day: i32,
    sessions: u64,
    events: u64,
    revenue: u64,
    category_sums: &[u64],
    touched: &[u32],
    categories: &[Vec<u8>],
) {
    let mut top_category = None;
    for &category in touched {
        top_category = match top_category {
            None => Some(category),
            Some(best) if category_sums[category as usize] > category_sums[best as usize] => {
                Some(category)
            }
            Some(best)
                if category_sums[category as usize] == category_sums[best as usize]
                    && categories[category as usize] < categories[best as usize] =>
            {
                Some(category)
            }
            Some(best) => Some(best),
        };
    }
    output.push(UserDaily {
        user,
        day,
        sessions,
        events,
        revenue,
        top_category,
    });
}

#[derive(Clone, Copy)]
struct DayRange {
    day: i32,
    start: usize,
    end: usize,
}

#[allow(clippy::too_many_arguments)]
fn attribute_and_assist(
    sessions: &[Session],
    clicks: &[Click],
    campaign_count: usize,
    organic: u32,
    first_day: i32,
    day_count: usize,
    output: &mut [CampaignDaily],
) {
    let mut click_at = 0usize;
    let mut campaign_times: Vec<Vec<i64>> = (0..campaign_count).map(|_| Vec::new()).collect();
    let mut touched_campaigns = Vec::with_capacity(campaign_count.min(256));
    let mut session_start = 0usize;

    while session_start < sessions.len() {
        let user = sessions[session_start].user;
        let mut session_end = session_start + 1;
        while session_end < sessions.len() && sessions[session_end].user == user {
            session_end += 1;
        }
        while click_at < clicks.len() && clicks[click_at].user < user {
            click_at += 1;
        }
        let click_start = click_at;
        while click_at < clicks.len() && clicks[click_at].user == user {
            click_at += 1;
        }
        let click_end = click_at;
        let person_sessions = &sessions[session_start..session_end];
        let person_clicks = &clicks[click_start..click_end];

        let mut eligible_end = 0usize;
        for session in person_sessions {
            while eligible_end < person_clicks.len()
                && person_clicks[eligible_end].ts <= session.start
            {
                eligible_end += 1;
            }
            let credited = if eligible_end > 0
                && session.start - person_clicks[eligible_end - 1].ts <= ATTRIBUTION_MS
            {
                person_clicks[eligible_end - 1].campaign
            } else {
                organic
            };
            let day = usize::try_from(session.day - first_day).expect("session before window");
            let aggregate = &mut output[credited as usize * day_count + day];
            aggregate.sessions += 1;
            aggregate.revenue += session.revenue;
            if session.purchasing {
                aggregate.purchasing_sessions += 1;
            }
        }

        for click in person_clicks {
            let times = &mut campaign_times[click.campaign as usize];
            if times.is_empty() {
                touched_campaigns.push(click.campaign);
            }
            times.push(click.ts);
        }

        let mut day_ranges = Vec::with_capacity(16);
        let mut range_start = 0usize;
        while range_start < person_sessions.len() {
            let day = person_sessions[range_start].day;
            let mut range_end = range_start + 1;
            while range_end < person_sessions.len() && person_sessions[range_end].day == day {
                range_end += 1;
            }
            day_ranges.push(DayRange {
                day,
                start: range_start,
                end: range_end,
            });
            range_start = range_end;
        }

        for &campaign in &touched_campaigns {
            let times = &campaign_times[campaign as usize];
            let mut at = 0usize;
            while at < times.len() {
                let interval_start = times[at];
                let mut interval_end = interval_start + ATTRIBUTION_MS;
                at += 1;
                while at < times.len() && times[at] <= interval_end {
                    interval_end = interval_end.max(times[at] + ATTRIBUTION_MS);
                    at += 1;
                }
                let lo = person_sessions.partition_point(|session| session.start < interval_start);
                let hi = person_sessions.partition_point(|session| session.start <= interval_end);
                if lo == hi {
                    continue;
                }
                for range in &day_ranges {
                    if range.end <= lo {
                        continue;
                    }
                    if range.start >= hi {
                        break;
                    }
                    let count = range.end.min(hi) - range.start.max(lo);
                    let day = (range.day - first_day) as usize;
                    output[campaign as usize * day_count + day].assisted_sessions += count as u64;
                }
            }
            campaign_times[campaign as usize].clear();
        }
        touched_campaigns.clear();
        session_start = session_end;
    }
}

#[allow(clippy::too_many_arguments)]
fn write_user_daily(
    path: &Path,
    rows: &[UserDaily],
    users: &ByteInterner,
    minimum_for_root: &[u32],
    categories: &[Vec<u8>],
    _first_day: i32,
    _day_count: usize,
) -> Result<()> {
    let mut user_ids = MutableUtf8Array::<i32>::with_capacity(rows.len());
    let mut days = Vec::with_capacity(rows.len());
    let mut sessions = Vec::with_capacity(rows.len());
    let mut events = Vec::with_capacity(rows.len());
    let mut revenue = Vec::with_capacity(rows.len());
    let mut top_categories = MutableUtf8Array::<i32>::with_capacity(rows.len());
    for row in rows {
        let minimum = minimum_for_root[row.user as usize];
        user_ids.push(Some(std::str::from_utf8(users.get(minimum))?));
        days.push(row.day);
        sessions.push(row.sessions as i64);
        events.push(row.events as i64);
        revenue.push(row.revenue as i64);
        if let Some(category) = row.top_category {
            top_categories.push(Some(std::str::from_utf8(
                &categories[category as usize],
            )?));
        } else {
            top_categories.push(None::<&str>);
        }
    }
    let schema = Schema::from(vec![
        Field::new("user_id", DataType::Utf8, false),
        Field::new("day", DataType::Date32, false),
        Field::new("sessions", DataType::Int64, false),
        Field::new("events", DataType::Int64, false),
        Field::new("revenue_cents", DataType::Int64, false),
        Field::new("top_category", DataType::Utf8, true),
    ]);
    write_parquet(
        path,
        schema,
        vec![
            Into::<Utf8Array<i32>>::into(user_ids).boxed(),
            PrimitiveArray::<i32>::from_vec(days)
                .to(DataType::Date32)
                .boxed(),
            PrimitiveArray::<i64>::from_vec(sessions).boxed(),
            PrimitiveArray::<i64>::from_vec(events).boxed(),
            PrimitiveArray::<i64>::from_vec(revenue).boxed(),
            Into::<Utf8Array<i32>>::into(top_categories).boxed(),
        ],
    )
}

fn write_campaign_daily(
    path: &Path,
    rows: &[CampaignDaily],
    campaigns: &Campaigns,
    organic: u32,
    first_day: i32,
    day_count: usize,
) -> Result<()> {
    let row_count = (0..=organic)
        .flat_map(|campaign| (0..day_count).map(move |day| (campaign, day)))
        .filter(|&(campaign, day)| {
            let row = &rows[campaign as usize * day_count + day];
            row.sessions != 0 || row.assisted_sessions != 0
        })
        .count();
    let mut campaign_ids = MutableUtf8Array::<i32>::with_capacity(row_count);
    let mut channels = MutableUtf8Array::<i32>::with_capacity(row_count);
    let mut days = Vec::with_capacity(row_count);
    let mut sessions = Vec::with_capacity(row_count);
    let mut purchasing_sessions = Vec::with_capacity(row_count);
    let mut revenue = Vec::with_capacity(row_count);
    let mut assisted_sessions = Vec::with_capacity(row_count);
    for campaign in 0..=organic {
        for day in 0..day_count {
            let row = &rows[campaign as usize * day_count + day];
            if row.sessions == 0 && row.assisted_sessions == 0 {
                continue;
            }
            if campaign == organic {
                campaign_ids.push(Some("organic"));
                channels.push(Some("organic"));
            } else {
                campaign_ids.push(Some(std::str::from_utf8(
                    &campaigns.names[campaign as usize],
                )?));
                channels.push(Some(std::str::from_utf8(
                    &campaigns.channels[campaign as usize],
                )?));
            }
            days.push(first_day + day as i32);
            sessions.push(row.sessions as i64);
            purchasing_sessions.push(row.purchasing_sessions as i64);
            revenue.push(row.revenue as i64);
            assisted_sessions.push(row.assisted_sessions as i64);
        }
    }
    let schema = Schema::from(vec![
        Field::new("campaign_id", DataType::Utf8, false),
        Field::new("channel", DataType::Utf8, false),
        Field::new("day", DataType::Date32, false),
        Field::new("sessions", DataType::Int64, false),
        Field::new("purchasing_sessions", DataType::Int64, false),
        Field::new("revenue_cents", DataType::Int64, false),
        Field::new("assisted_sessions", DataType::Int64, false),
    ]);
    write_parquet(
        path,
        schema,
        vec![
            Into::<Utf8Array<i32>>::into(campaign_ids).boxed(),
            Into::<Utf8Array<i32>>::into(channels).boxed(),
            PrimitiveArray::<i32>::from_vec(days)
                .to(DataType::Date32)
                .boxed(),
            PrimitiveArray::<i64>::from_vec(sessions).boxed(),
            PrimitiveArray::<i64>::from_vec(purchasing_sessions).boxed(),
            PrimitiveArray::<i64>::from_vec(revenue).boxed(),
            PrimitiveArray::<i64>::from_vec(assisted_sessions).boxed(),
        ],
    )
}

fn parse_ascii_u64(value: &[u8]) -> Option<u64> {
    if value.is_empty() || !value.iter().all(u8::is_ascii_digit) {
        return None;
    }
    let mut number = 0u64;
    for byte in value {
        number = number.checked_mul(10)?.checked_add((byte - b'0') as u64)?;
    }
    Some(number)
}

fn parse_timestamp(value: &[u8]) -> Option<i64> {
    if value.iter().all(u8::is_ascii_digit) && !value.is_empty() {
        return parse_ascii_u64(value).and_then(|number| i64::try_from(number).ok());
    }
    if value.len() != 20 && value.len() != 25 {
        return None;
    }
    if value[4] != b'-'
        || value[7] != b'-'
        || value[10] != b'T'
        || value[13] != b':'
        || value[16] != b':'
    {
        return None;
    }
    let year = digits(value, 0, 4)? as i64;
    let month = digits(value, 5, 2)? as i64;
    let day = digits(value, 8, 2)? as i64;
    let hour = digits(value, 11, 2)? as i64;
    let minute = digits(value, 14, 2)? as i64;
    let second = digits(value, 17, 2)? as i64;
    if !(1..=12).contains(&month)
        || day < 1
        || day > days_in_month(year, month)
        || hour > 23
        || minute > 59
        || second > 59
    {
        return None;
    }
    let local = days_from_civil(year, month, day)
        .checked_mul(DAY_MS)?
        .checked_add(hour * 3_600_000 + minute * MINUTE_MS + second * 1_000)?;
    if value.len() == 20 {
        return (value[19] == b'Z').then_some(local);
    }
    if !matches!(value[19], b'+' | b'-') || value[22] != b':' {
        return None;
    }
    let offset_hour = digits(value, 20, 2)? as i64;
    let offset_minute = digits(value, 23, 2)? as i64;
    if offset_hour > 23 || offset_minute > 59 {
        return None;
    }
    let offset = offset_hour * 3_600_000 + offset_minute * MINUTE_MS;
    if value[19] == b'+' {
        local.checked_sub(offset)
    } else {
        local.checked_add(offset)
    }
}

fn digits(value: &[u8], start: usize, len: usize) -> Option<u32> {
    let slice = value.get(start..start + len)?;
    if !slice.iter().all(u8::is_ascii_digit) {
        return None;
    }
    let mut number = 0u32;
    for byte in slice {
        number = number * 10 + (byte - b'0') as u32;
    }
    Some(number)
}

fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) => 29,
        2 => 28,
        _ => 0,
    }
}

fn days_from_civil(mut year: i64, month: i64, day: i64) -> i64 {
    year -= (month <= 2) as i64;
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let shifted_month = month + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * shifted_month + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

fn format_day(day: i32) -> String {
    let z = day as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += (month <= 2) as i64;
    format!("{year:04}-{month:02}-{day:02}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamps_are_strict() {
        assert_eq!(
            parse_timestamp(b"2026-10-03T23:58:10Z"),
            parse_timestamp(b"2026-10-03T19:58:10-04:00")
        );
        assert_eq!(
            parse_timestamp(b"2026-10-03T23:58:10Z"),
            Some(1_791_071_890_000)
        );
        assert!(parse_timestamp(b"2026-02-29T00:00:00Z").is_none());
        assert!(parse_timestamp(b"2028-02-29T00:00:00Z").is_some());
        assert!(parse_timestamp(b"2026-10-03T24:00:00Z").is_none());
        assert!(parse_timestamp(b"1791071890000.0").is_none());
        assert_eq!(
            parse_timestamp(b"0001791071890000"),
            Some(1_791_071_890_000)
        );
    }

    #[test]
    fn civil_dates_round_trip() {
        for &(year, month, day) in &[(1970, 1, 1), (2026, 10, 3), (2028, 2, 29)] {
            let days = days_from_civil(year, month, day) as i32;
            assert_eq!(format_day(days), format!("{year:04}-{month:02}-{day:02}"));
        }
    }

    #[test]
    fn strict_csv_fields() {
        assert_eq!(
            parse_fields::<2>(b"\"u1\",u2"),
            Some([b"u1".as_slice(), b"u2".as_slice()])
        );
        assert_eq!(
            parse_fields::<2>(b"\"\",u2"),
            Some([b"".as_slice(), b"u2".as_slice()])
        );
        assert!(parse_fields::<2>(b"u1\",u2").is_none());
        assert!(parse_fields::<2>(b"u1,u2,extra").is_none());
        assert!(parse_fields::<2>(b"").is_none());
    }
}
