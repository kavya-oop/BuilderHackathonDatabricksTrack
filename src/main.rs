use rayon::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::hash::{BuildHasherDefault, Hasher};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use xxhash_rust::xxh3::xxh3_64;

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
    if value.len() < 2 || value.len() > 9 || !matches!(value[0], b'e' | b'u' | b'k') {
        return None;
    }
    let mut body = 0u64;
    for byte in &value[1..] {
        let digit = match byte {
            b'0'..=b'9' => byte - b'0',
            b'a'..=b'z' => byte - b'a' + 10,
            _ => return None,
        };
        body = (body << 6) | digit as u64;
    }
    Some(((value[0] as u64) << 56) | (((value.len() - 1) as u64) << 52) | body)
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

fn main() {
    if let Err(error) = run() {
        eprintln!("shopping-sessions: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let started = Instant::now();
    let data = PathBuf::from(env::var("DATA_DIR").unwrap_or_else(|_| "/data".into()));
    let scratch = PathBuf::from(env::var("SCRATCH_DIR").unwrap_or_else(|_| "/scratch".into()));
    fs::create_dir_all(&scratch)?;

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

    let link_bytes = total_size(&link_files)?;
    let mut users = ByteInterner::with_capacity(link_bytes / 20 + 1024, link_bytes / 2 + 1024);
    let mut dsu = Dsu::new();
    read_links(&link_files, &mut users, &mut dsu)?;
    eprintln!("links parsed in {:.3}s", started.elapsed().as_secs_f64());

    let event_bytes = total_size(&event_files)?;
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
        &scratch.join("user_daily.csv"),
        &user_daily,
        &users,
        &minimum_for_root,
        &categories,
        first_day,
        day_count,
    )?;
    write_campaign_daily(
        &scratch.join("campaign_daily.csv"),
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
    first_day: i32,
    day_count: usize,
) -> Result<()> {
    let dates: Vec<String> = (0..day_count)
        .map(|offset| format_day(first_day + offset as i32))
        .collect();
    let mut writer = BufWriter::with_capacity(1 << 20, File::create(path)?);
    writer.write_all(b"user_id,day,sessions,events,revenue_cents,top_category\n")?;
    for row in rows {
        let minimum = minimum_for_root[row.user as usize];
        write_csv_field(&mut writer, users.get(minimum))?;
        write!(
            writer,
            ",{},{},{},{},",
            dates[(row.day - first_day) as usize],
            row.sessions,
            row.events,
            row.revenue
        )?;
        if let Some(category) = row.top_category {
            write_csv_field(&mut writer, &categories[category as usize])?;
        }
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

fn write_campaign_daily(
    path: &Path,
    rows: &[CampaignDaily],
    campaigns: &Campaigns,
    organic: u32,
    first_day: i32,
    day_count: usize,
) -> Result<()> {
    let dates: Vec<String> = (0..day_count)
        .map(|offset| format_day(first_day + offset as i32))
        .collect();
    let mut writer = BufWriter::with_capacity(1 << 20, File::create(path)?);
    writer.write_all(
        b"campaign_id,channel,day,sessions,purchasing_sessions,revenue_cents,assisted_sessions\n",
    )?;
    for campaign in 0..=organic {
        for day in 0..day_count {
            let row = &rows[campaign as usize * day_count + day];
            if row.sessions == 0 && row.assisted_sessions == 0 {
                continue;
            }
            if campaign == organic {
                writer.write_all(b"organic,organic")?;
            } else {
                write_csv_field(&mut writer, &campaigns.names[campaign as usize])?;
                writer.write_all(b",")?;
                write_csv_field(&mut writer, &campaigns.channels[campaign as usize])?;
            }
            write!(
                writer,
                ",{},{},{},{},{}\n",
                dates[day],
                row.sessions,
                row.purchasing_sessions,
                row.revenue,
                row.assisted_sessions
            )?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn write_csv_field(writer: &mut impl Write, value: &[u8]) -> Result<()> {
    if value
        .iter()
        .any(|byte| matches!(*byte, b',' | b'"' | b'\n' | b'\r'))
    {
        writer.write_all(b"\"")?;
        for byte in value {
            if *byte == b'"' {
                writer.write_all(b"\"\"")?;
            } else {
                writer.write_all(std::slice::from_ref(byte))?;
            }
        }
        writer.write_all(b"\"")?;
    } else {
        writer.write_all(value)?;
    }
    Ok(())
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
