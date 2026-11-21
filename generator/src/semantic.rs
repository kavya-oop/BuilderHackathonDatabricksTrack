//! A byte-free replay of the published generator. It consumes the identical
//! random stream, including CSV quoting, malformation and part-selection
//! draws, while retaining only rows that can affect the required answer.

use crate::plan::{Dataset, UserPlan, SLOTS, SLOTS_PER_DAY, WINDOW_DAYS};
use crate::random::{chance_threshold, Random, Stream};
use crate::text::{scramble, DAY, HOUR, MINUTE, SECOND};
use std::cmp::Ordering;

pub const ALIAS_SLOTS: u64 = 512;
/// Normal generated people are represented by their dense pre-scramble alias
/// number. Only case-variant IDs need the packed byte representation during
/// replay; the high bit tags those rare values.
pub const EXCEPTIONAL_PERSON: u64 = 1 << 63;
const KIOSK_BASE: u64 = 1 << 39;
const FIRST_ALIAS_SHARE: f64 = 0.5;
const REPEATED_LINK_SHARE: f64 = 0.08;
const SELF_LINK_SHARE: f64 = 0.01;
const UNUSED_ALIAS_SHARE: f64 = 0.05;
const LINK_CASE_VARIANT: f64 = 0.000_5;
const SLOT: i64 = 2 * HOUR;
const LATEST_START: i64 = 25 * MINUTE;
const LONGEST_SESSION: i64 = 30 * MINUTE - SECOND;
const SESSION_GAP: i64 = 30 * MINUTE;
const MIDNIGHT_SHARE: f64 = 0.025;
const BOUNDARY_GAP_SHARE: f64 = 0.001;
const MILLISECOND_START_SHARE: f64 = 0.03;
const TARGETED_SESSION_SHARE: f64 = 0.45;
const WHALE_CAMPAIGN_SHARE: f64 = 0.3;
const LAST_SLOT_OF_DAY: usize = SLOTS_PER_DAY - 1;
const MOST_SESSIONS_TO_CROSS: usize = 80;
const BAD_ORIGINAL_SHARE: f64 = 0.02;
const ODD_PRICE_SHARE: f64 = 0.001;
const SAME_MILLISECOND_SHARE: f64 = 0.01;
const CLICK_REACH: i64 = 7 * DAY;
const FIRST_ALIAS_THRESHOLD: u64 = chance_threshold(FIRST_ALIAS_SHARE);
const MIDNIGHT_THRESHOLD: u64 = chance_threshold(MIDNIGHT_SHARE);
const BOUNDARY_GAP_THRESHOLD: u64 = chance_threshold(BOUNDARY_GAP_SHARE);
const MILLISECOND_START_THRESHOLD: u64 = chance_threshold(MILLISECOND_START_SHARE);
const ODD_PRICE_THRESHOLD: u64 = chance_threshold(ODD_PRICE_SHARE);
const BAD_ORIGINAL_THRESHOLD: u64 = chance_threshold(BAD_ORIGINAL_SHARE);
const TARGETED_SESSION_THRESHOLD: u64 = chance_threshold(TARGETED_SESSION_SHARE);
const SAME_MILLISECOND_THRESHOLD: u64 = chance_threshold(SAME_MILLISECOND_SHARE);
const WHALE_CAMPAIGN_THRESHOLD: u64 = chance_threshold(WHALE_CAMPAIGN_SHARE);
const UNUSED_ALIAS_THRESHOLD: u64 = chance_threshold(UNUSED_ALIAS_SHARE);
const SELF_LINK_THRESHOLD: u64 = chance_threshold(SELF_LINK_SHARE);
const REPEATED_LINK_THRESHOLD: u64 = chance_threshold(REPEATED_LINK_SHARE);
const HALF_THRESHOLD: u64 = chance_threshold(0.5);
const LINK_CASE_VARIANT_THRESHOLD: u64 = chance_threshold(LINK_CASE_VARIANT);

#[derive(Clone, Copy)]
pub struct SemanticEvent {
    pub ts: i64,
    pub person: u64,
    pub price: u32,
    pub product: u32,
    pub purchase: bool,
}

#[derive(Clone, Copy)]
pub struct SemanticClick {
    pub ts: i64,
    pub person: u64,
    pub campaign: u32,
    pub id: u64,
}

pub struct Shard {
    pub events: Vec<SemanticEvent>,
    pub clicks: Vec<SemanticClick>,
    pub links: Vec<(u64, u64)>,
    event_times: Vec<i64>,
    gaps: Vec<i64>,
}

impl Shard {
    fn new(event_capacity: usize, click_capacity: usize) -> Self {
        Self {
            events: Vec::with_capacity(event_capacity),
            clicks: Vec::with_capacity(click_capacity),
            links: Vec::new(),
            event_times: Vec::new(),
            gaps: Vec::new(),
        }
    }

    fn start_user(&mut self) {
        self.events.clear();
        self.clicks.clear();
        self.links.clear();
    }
}

/// Replays one generator shard while exposing each completed user's compact
/// semantic rows to the caller. The buffers are cleared and reused after the
/// callback, so the normal path never retains the full synthetic dataset.
pub fn generate_shard_with<F>(
    dataset: &Dataset,
    users: std::ops::Range<usize>,
    mut consume: F,
) where
    F: FnMut(usize, UserPlan, &mut Shard),
{
    let mut shard = Shard::new(0, 0);
    for index in users {
        let plan = dataset.users[index];
        if plan.events == 0 {
            continue;
        }
        shard.start_user();
        SemanticUser::new(dataset, index, plan).generate(&mut shard);
        consume(index, plan, &mut shard);
    }
}

pub fn generate_shard(dataset: &Dataset, users: std::ops::Range<usize>) -> Shard {
    let event_capacity = users
        .clone()
        .map(|index| dataset.users[index].events as usize)
        .sum();
    let click_capacity = users
        .clone()
        .map(|index| dataset.users[index].clicks as usize)
        .sum();
    let mut shard = Shard::new(event_capacity, click_capacity);
    for index in users {
        let plan = dataset.users[index];
        if plan.events > 0 {
            SemanticUser::new(dataset, index, plan).generate(&mut shard);
        }
    }
    shard
}

struct SemanticUser<'a> {
    dataset: &'a Dataset,
    plan: UserPlan,
    index: usize,
    random: Random,
    id_count: usize,
    session_starts: [i64; SLOTS as usize],
    session_count: usize,
}

impl<'a> SemanticUser<'a> {
    fn new(dataset: &'a Dataset, index: usize, plan: UserPlan) -> Self {
        let id_count = plan.aliases.max(1) as usize;
        Self {
            dataset,
            plan,
            index,
            random: Random::new(dataset.seed, Stream::User, index as u64),
            id_count,
            session_starts: [0; SLOTS as usize],
            session_count: 0,
        }
    }

    fn generate(&mut self, shard: &mut Shard) {
        self.layout(shard);
        let mut event_index = self.plan.first_event;
        for at in 0..shard.event_times.len() {
            let ts = shard.event_times[at];
            self.event(shard, event_index, ts);
            event_index += 1;
        }
        self.clicks(shard);
        self.links(shard);
    }

    fn alias(random: &mut Random, plan: &UserPlan, aliases: usize) -> usize {
        if aliases == 1 || (!plan.chain && random.chance_with_threshold(FIRST_ALIAS_THRESHOLD)) {
            0
        } else {
            random.below(aliases as u64) as usize
        }
    }

    fn user_field(&mut self, number: u64) -> Option<u64> {
        if self
            .random
            .chance_with_threshold(self.dataset.rate_thresholds.bad_user)
        {
            None
        } else if self
            .random
            .chance_with_threshold(self.dataset.rate_thresholds.case_variant)
        {
            Some(EXCEPTIONAL_PERSON | uppercase_packed(user_id(self.dataset, number)))
        } else {
            Some(number)
        }
    }

    fn layout(&mut self, shard: &mut Shard) {
        let random = &mut self.random;
        let count = self.plan.sessions as usize;
        let mut used = [false; SLOTS as usize];
        let mut slots = [(0usize, false); SLOTS as usize];
        let mut slot_count = 0usize;
        if count == SLOTS as usize {
            for (slot, target) in slots[..count].iter_mut().enumerate() {
                *target = (slot, false);
            }
            slot_count = count;
        } else {
            let may_cross = count <= MOST_SESSIONS_TO_CROSS;
            for _ in 0..count {
                if may_cross && random.chance_with_threshold(MIDNIGHT_THRESHOLD) {
                    let day = random.below(WINDOW_DAYS as u64 - 1) as usize;
                    let slot = day * SLOTS_PER_DAY + LAST_SLOT_OF_DAY;
                    if !used[slot] && !used[slot + 1] {
                        used[slot] = true;
                        used[slot + 1] = true;
                        slots[slot_count] = (slot, true);
                        slot_count += 1;
                        continue;
                    }
                }
                loop {
                    let slot = random.below(u64::from(SLOTS)) as usize;
                    if !used[slot] {
                        used[slot] = true;
                        slots[slot_count] = (slot, false);
                        slot_count += 1;
                        break;
                    }
                }
            }
            slots[..slot_count].sort_unstable();
        }

        let mut sizes = [0u64; SLOTS as usize];
        for size in &mut sizes[..count] {
            *size = 1 + random.below(4);
        }
        let weight_total: u64 = sizes[..count].iter().sum();
        let extra = self.plan.events - count as u64;
        let mut given = 0;
        let mut running = 0;
        for size in &mut sizes[..count] {
            let weight = *size;
            running += weight;
            let through = extra * running / weight_total;
            *size = 1 + through - given;
            given = through;
        }

        shard.event_times.clear();
        shard.event_times.reserve(self.plan.events as usize);
        self.session_count = count;
        for (session_index, (&(slot, crossing), &size)) in slots[..slot_count]
            .iter()
            .zip(&sizes[..count])
            .enumerate()
        {
            shard.gaps.clear();
            shard.gaps.reserve(size.saturating_sub(1) as usize);
            shard
                .gaps
                .extend((1..size).map(|_| random.spread(10) * SECOND));
            let natural: i64 = shard.gaps.iter().sum();
            if natural > LONGEST_SESSION {
                for gap in &mut shard.gaps {
                    *gap = *gap * LONGEST_SESSION / natural / SECOND * SECOND;
                }
            }
            if let Some(at) = shard
                .gaps
                .iter()
                .position(|_| random.chance_with_threshold(BOUNDARY_GAP_THRESHOLD))
            {
                shard.gaps[at] = SESSION_GAP + random.between(0, 1);
            }
            let duration: i64 = shard.gaps.iter().sum();
            let slot_start = self.dataset.window_start + slot as i64 * SLOT;
            let start = if crossing && duration >= 2 * SECOND {
                slot_start + SLOT - random.between(1, duration / SECOND) * SECOND
            } else {
                slot_start + random.below((LATEST_START / SECOND) as u64) as i64 * SECOND
            } + if random.chance_with_threshold(MILLISECOND_START_THRESHOLD) {
                random.between(1, 999)
            } else {
                0
            };
            self.session_starts[session_index] = start;
            let mut ts = start;
            shard.event_times.push(ts);
            for &gap in &shard.gaps {
                ts += gap;
                shard.event_times.push(ts);
            }
        }
    }

    fn event(&mut self, shard: &mut Shard, index: u64, ts: i64) {
        self.random.skip(1); // event ID case; duplicates keep the same ID
        let alias = Self::alias(&mut self.random, &self.plan, self.id_count);
        let number = self.index as u64 * ALIAS_SLOTS + alias as u64;
        let mut person = self.user_field(number);

        let timestamp_is_bad = self
            .random
            .chance_with_threshold(self.dataset.rate_thresholds.bad_timestamp);
        if timestamp_is_bad {
            let outside = outside(
                &mut self.random,
                self.dataset.window_start,
                self.dataset.window_end(),
            );
            consume_bad_timestamp(&mut self.random, outside);
        } else {
            consume_valid_timestamp(&mut self.random, ts);
        }
        let (original_on_time, duplicate_on_time) = ingest_validity(
            &mut self.random,
            self.dataset.rate_thresholds.late,
            ts,
        );

        let purchase = self.random.below_lt(100, 3);
        let type_valid = if self
            .random
            .chance_with_threshold(self.dataset.rate_thresholds.bad_type)
        {
            self.random.skip(1);
            false
        } else {
            true
        };
        let selected_product = if purchase {
            self.dataset.product_by_rank[self
                .random
                .by_cumulative(&self.dataset.product_rank_cumulative)]
                as usize
        } else {
            // Product choice cannot affect a non-purchase row, but its draw
            // still belongs to the deterministic stream.
            self.random.skip(1);
            0
        };
        let (product, product_present) = if self
            .random
            .chance_with_threshold(self.dataset.rate_thresholds.unknown_product)
        {
            match self.random.below(10) {
                0 => (u32::MAX, false),
                1 => (u32::MAX, true),
                _ => {
                    self.random.skip(1);
                    (u32::MAX, true)
                }
            }
        } else {
            (selected_product as u32, true)
        };
        let (price, price_valid) = if purchase {
            if self
                .random
                .chance_with_threshold(self.dataset.rate_thresholds.bad_price)
            {
                self.random.skip(1);
                (0, false)
            } else if self.random.chance_with_threshold(ODD_PRICE_THRESHOLD) {
                let price = [10_000_000, 1, 250][self.random.below(3) as usize];
                (price, true)
            } else {
                (self.dataset.product_price[selected_product] as u32, true)
            }
        } else {
            if !self.random.below_lt(20, 19) {
                self.random.skip(1);
            }
            (0, true)
        };
        let base_valid =
            person.is_some() && !timestamp_is_bad && type_valid && product_present && price_valid;

        let copies = if self
            .random
            .chance_with_threshold(self.dataset.rate_thresholds.event_duplicates)
        {
            1 + self.random.below(3)
        } else {
            0
        };
        let special_bad_original =
            copies > 0
                && !timestamp_is_bad
                && self.random.chance_with_threshold(BAD_ORIGINAL_THRESHOLD);
        if special_bad_original {
            let outside = outside(
                &mut self.random,
                self.dataset.window_start,
                self.dataset.window_end(),
            );
            consume_bad_timestamp(&mut self.random, outside);
        }
        let original_present = emit(
            &mut self.random,
            self.dataset.rate_thresholds.malformed,
            7,
        );
        let mut winner_recorded = original_present;
        if original_present && base_valid && !special_bad_original && original_on_time {
            shard.events.push(SemanticEvent {
                ts,
                person: person.unwrap(),
                price,
                product,
                purchase,
            });
        }
        if original_present && copies > 0 && scramble(index, self.dataset.click_key) % 64 == 0 {
            let other = (self.index + 1) % self.dataset.users.len();
            person = Some(other as u64 * ALIAS_SLOTS);
        }
        for _ in 0..copies {
            // The exact duplicate ingest timestamp cannot affect the answer;
            // only whether it remains within the one-day limit matters.
            self.random.skip(1);
            let present = emit(
                &mut self.random,
                self.dataset.rate_thresholds.malformed,
                7,
            );
            if present && !winner_recorded {
                winner_recorded = true;
                if base_valid && duplicate_on_time {
                    shard.events.push(SemanticEvent {
                        ts,
                        person: person.unwrap(),
                        price,
                        product,
                        purchase,
                    });
                }
            }
        }
    }

    fn clicks(&mut self, shard: &mut Shard) {
        let earliest = self.dataset.window_start - CLICK_REACH;
        let session_starts = &self.session_starts[..self.session_count];
        let mut targets = [0i64; SLOTS as usize];
        let mut target_count = 0usize;
        for &start in session_starts {
            if self
                .random
                .chance_with_threshold(TARGETED_SESSION_THRESHOLD)
            {
                targets[target_count] = start;
                target_count += 1;
            }
        }
        if target_count == 0 {
            targets[0] = *self.random.pick(session_starts);
            target_count = 1;
        }
        let mut index = self.plan.first_click;
        let end = index + self.plan.clicks;
        while index < end {
            let start = *self.random.pick(&targets[..target_count]);
            let delta_draw = self.random.next_u64();
            let delta = if Random::bucket_lt(delta_draw, 1000, 4) {
                CLICK_REACH
            } else if Random::bucket_lt(delta_draw, 1000, 8) {
                CLICK_REACH + 1
            } else if Random::bucket_lt(delta_draw, 1000, 12) {
                0
            } else if Random::bucket_lt(delta_draw, 1000, 16) {
                -1
            } else if Random::bucket_lt(delta_draw, 1000, 750) {
                (self.random.spread(19) * SECOND).min(CLICK_REACH)
            } else {
                CLICK_REACH + self.random.between(1, 3 * DAY / SECOND) * SECOND
            };
            let mut ts = start - delta;
            if ts < earliest {
                ts = start
                    - self.random.below(((start - earliest) / SECOND) as u64 + 1) as i64 * SECOND;
            }
            let same_millisecond = end - index >= 2
                && self.random.chance_with_threshold(SAME_MILLISECOND_THRESHOLD);
            for _ in 0..1 + u64::from(same_millisecond) {
                self.click(shard, index, ts);
                index += 1;
            }
        }
    }

    fn click(&mut self, shard: &mut Shard, index: u64, ts: i64) {
        let id_case_variant = self
            .random
            .chance_with_threshold(self.dataset.rate_thresholds.case_variant);
        // Preserve the scrambled value instead of formatting base-36 for every
        // click. The high bit records the rare upper-case variant; formatting
        // order is reconstructed only when equal timestamps need an ID tie-break.
        let id = scramble(index, self.dataset.click_key)
            | if id_case_variant { 1 << 63 } else { 0 };
        let alias = Self::alias(&mut self.random, &self.plan, self.id_count);
        let number = self.index as u64 * ALIAS_SLOTS + alias as u64;
        let person = self.user_field(number);
        let campaign = if self
            .random
            .chance_with_threshold(self.dataset.rate_thresholds.unknown_campaign)
        {
            // The upper-case choice and the generated unknown name only affect
            // bytes; unknown campaigns are discarded either way.
            self.random.skip(2);
            None
        } else if self.random.chance_with_threshold(WHALE_CAMPAIGN_THRESHOLD) {
            Some(*self.random.pick(&self.dataset.whale_campaigns))
        } else {
            Some(
                self.dataset.campaign_by_rank[self
                    .random
                    .by_cumulative(&self.dataset.campaign_rank_cumulative)],
            )
        };
        let timestamp_is_bad = self
            .random
            .chance_with_threshold(self.dataset.rate_thresholds.bad_timestamp);
        if timestamp_is_bad {
            let outside = outside(
                &mut self.random,
                self.dataset.window_start - CLICK_REACH,
                self.dataset.window_end(),
            );
            consume_bad_timestamp(&mut self.random, outside);
        } else {
            consume_valid_timestamp(&mut self.random, ts);
        }
        let copies = if self
            .random
            .chance_with_threshold(self.dataset.rate_thresholds.click_duplicates)
        {
            1 + self.random.below(2)
        } else {
            0
        };
        let mut winner_recorded = false;
        for _ in 0..=copies {
            let present = emit(
                &mut self.random,
                self.dataset.rate_thresholds.malformed,
                4,
            );
            if present && !winner_recorded {
                winner_recorded = true;
                if let (Some(person), Some(campaign)) = (person, campaign) {
                    if !timestamp_is_bad
                        && ts >= self.dataset.window_start - CLICK_REACH
                        && ts < self.dataset.window_end()
                    {
                        shard.clicks.push(SemanticClick {
                            ts,
                            person,
                            campaign,
                            id,
                        });
                    }
                }
            }
        }
    }

    fn links(&mut self, shard: &mut Shard) {
        let count = self.id_count;
        let base = self.index as u64 * ALIAS_SLOTS;
        let mut pairs = [(0u64, 0u64); ALIAS_SLOTS as usize + 3];
        let mut pair_count = 0usize;
        if self.plan.chain {
            let mut order = [0usize; ALIAS_SLOTS as usize];
            for (index, target) in order[..count].iter_mut().enumerate() {
                *target = index;
            }
            self.random.shuffle(&mut order[..count]);
            for pair in order[..count].windows(2) {
                pairs[pair_count] = (base + pair[0] as u64, base + pair[1] as u64);
                pair_count += 1;
            }
        } else {
            for alias in 1..count {
                let earlier = self.random.below(alias as u64) as usize;
                pairs[pair_count] = (base + alias as u64, base + earlier as u64);
                pair_count += 1;
            }
        }
        if self.random.chance_with_threshold(UNUSED_ALIAS_THRESHOLD) {
            let unused = base + count as u64;
            pairs[pair_count] = (base + self.random.below(count as u64), unused);
            pair_count += 1;
        }
        if self.plan.kiosk > 0 {
            let kiosk = KIOSK_BASE + u64::from(self.plan.kiosk);
            pairs[pair_count] = (base + self.random.below(count as u64), kiosk);
            pair_count += 1;
        }
        if self.random.chance_with_threshold(SELF_LINK_THRESHOLD) {
            let id = base + self.random.below(count as u64);
            pairs[pair_count] = (id, id);
            pair_count += 1;
        }
        for &(left, right) in &pairs[..pair_count] {
            let copies = 1
                + usize::from(
                    self.random
                        .chance_with_threshold(REPEATED_LINK_THRESHOLD),
                );
            for _ in 0..copies {
                let (first, second) = if self.random.chance_with_threshold(HALF_THRESHOLD) {
                    (left, right)
                } else {
                    (right, left)
                };
                let first = self.link_field(first);
                let second = self.link_field(second);
                if emit(
                    &mut self.random,
                    self.dataset.rate_thresholds.malformed,
                    2,
                ) {
                    if let (Some(first), Some(second)) = (first, second) {
                        shard.links.push((first, second));
                    }
                }
            }
        }
    }

    fn link_field(&mut self, number: u64) -> Option<u64> {
        if self
            .random
            .chance_with_threshold(self.dataset.rate_thresholds.bad_user)
        {
            None
        } else if self
            .random
            .chance_with_threshold(LINK_CASE_VARIANT_THRESHOLD)
        {
            Some(EXCEPTIONAL_PERSON | uppercase_packed(user_id(self.dataset, number)))
        } else {
            Some(number)
        }
    }
}

pub fn user_id(dataset: &Dataset, number: u64) -> u64 {
    packed_id(b'u', scramble(number, dataset.user_key), false)
}

/// Convert the dense/exceptional person representation used by semantic replay
/// to the packed ASCII ordering used by output and canonical-ID selection.
pub fn packed_person(dataset: &Dataset, person: u64) -> u64 {
    if person & EXCEPTIONAL_PERSON == 0 {
        user_id(dataset, person)
    } else {
        person & !EXCEPTIONAL_PERSON
    }
}

/// Compare semantic click IDs in their emitted ASCII order without eagerly
/// building the `k<base36>` byte string for every generated click.
pub fn compare_click_ids(left: u64, right: u64) -> Ordering {
    let left_upper = left >> 63;
    let right_upper = right >> 63;
    match (left_upper, right_upper) {
        (1, 0) => return Ordering::Less,
        (0, 1) => return Ordering::Greater,
        _ => {}
    }

    fn digits(mut value: u64, out: &mut [u8; 8]) -> usize {
        let mut at = out.len();
        loop {
            at -= 1;
            out[at] = (value % 36) as u8;
            value /= 36;
            if value == 0 {
                return at;
            }
        }
    }

    let mut left_digits = [0u8; 8];
    let mut right_digits = [0u8; 8];
    let left_at = digits(left & ((1 << 40) - 1), &mut left_digits);
    let right_at = digits(right & ((1 << 40) - 1), &mut right_digits);
    left_digits[left_at..].cmp(&right_digits[right_at..])
}

fn packed_id(prefix: u8, value: u64, uppercase: bool) -> u64 {
    let mut reverse = [0u8; 8];
    let mut length = 0usize;
    let mut rest = value;
    loop {
        reverse[length] = (rest % 36) as u8;
        length += 1;
        rest /= 36;
        if rest == 0 {
            break;
        }
    }
    let mut packed = u64::from(if uppercase {
        prefix.to_ascii_uppercase()
    } else {
        prefix
    }) << 56;
    for index in 0..length {
        let digit = reverse[length - index - 1];
        let rank = if digit < 10 {
            digit + 1
        } else if uppercase {
            digit + 1
        } else {
            digit + 27
        };
        packed |= u64::from(rank) << (50 - index * 6);
    }
    packed
}

fn uppercase_packed(packed: u64) -> u64 {
    let mut output = u64::from(((packed >> 56) as u8).to_ascii_uppercase()) << 56;
    for index in 0..8 {
        let rank = ((packed >> (50 - index * 6)) & 63) as u8;
        if rank == 0 {
            break;
        }
        let upper = if rank >= 37 { rank - 26 } else { rank };
        output |= u64::from(upper) << (50 - index * 6);
    }
    output
}

fn consume_valid_timestamp(random: &mut Random, milliseconds: i64) {
    let draw = random.next_u64();
    if milliseconds % SECOND == 0
        && !Random::bucket_lt(draw, 10, 6)
        && Random::bucket_lt(draw, 10, 7)
    {
        random.skip(1);
    }
}

fn consume_bad_timestamp(random: &mut Random, outside: i64) {
    match random.below(15) {
        7 => {
            random.skip(1);
        }
        14 => consume_valid_timestamp(random, outside),
        _ => {}
    }
}

fn outside(random: &mut Random, window_start: i64, window_end: i64) -> i64 {
    match random.below(6) {
        0 => window_start - 1,
        1 => window_start - SECOND,
        2 => window_end,
        3 => window_start - random.between(1, 3 * DAY),
        _ => window_end + random.between(0, 3 * DAY),
    }
}

fn ingest_validity(random: &mut Random, late_threshold: u64, ts: i64) -> (bool, bool) {
    let floor = ts - ts.rem_euclid(SECOND);
    if random.chance_with_threshold(late_threshold) {
        let branch = random.next_u64();
        if floor == ts && Random::bucket_lt(branch, 10, 1) {
            // Exactly one day late admits the original but not a later copy.
            (true, false)
        } else if floor == ts && Random::bucket_lt(branch, 10, 2) {
            (false, false)
        } else {
            random.skip(1);
            (false, false)
        }
    } else {
        // Early-ingest formatting consumes one draw and its timestamp consumes
        // one more, but neither value affects semantic validity here.
        random.skip(2);
        (true, true)
    }
}

fn emit(random: &mut Random, malformed_threshold: u64, fields: usize) -> bool {
    // Quoting and the chosen output part affect bytes only. Advance the
    // SplitMix counter without evaluating those discarded random values.
    random.skip(fields);
    let mut present = true;
    if random.chance_with_threshold(malformed_threshold) {
        let fault = random.below(6);
        present = fault == 5;
        if fault == 5 {
            random.skip(1);
        }
    }
    random.skip(1);
    present
}
