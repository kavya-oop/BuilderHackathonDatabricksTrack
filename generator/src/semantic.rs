//! A byte-free replay of the published generator. It consumes the identical
//! random stream, including CSV quoting, malformation and part-selection
//! draws, while retaining only rows that can affect the required answer.

use crate::plan::{
    Dataset, Rates, UserPlan, CAMPAIGNS, PRODUCTS, SLOTS, SLOTS_PER_DAY, WINDOW_DAYS,
};
use crate::random::{Random, Stream};
use crate::text::{scramble, DAY, HOUR, MINUTE, SECOND};

const ALIAS_SLOTS: u64 = 512;
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
const UPPERCASE_CAMPAIGN_SHARE: f64 = 0.1;
const EARLY_INGEST_SHARE: f64 = 0.005;
const SAME_MILLISECOND_SHARE: f64 = 0.01;
const CLICK_REACH: i64 = 7 * DAY;

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
    let mut shard = Shard {
        events: Vec::with_capacity(event_capacity),
        clicks: Vec::with_capacity(click_capacity),
        links: Vec::new(),
    };
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
    ids: Vec<u64>,
}

impl<'a> SemanticUser<'a> {
    fn new(dataset: &'a Dataset, index: usize, plan: UserPlan) -> Self {
        let ids = (0..u64::from(plan.aliases.max(1)))
            .map(|alias| user_id(dataset, index as u64 * ALIAS_SLOTS + alias))
            .collect();
        Self {
            dataset,
            plan,
            index,
            random: Random::new(dataset.seed, Stream::User, index as u64),
            ids,
        }
    }

    fn generate(&mut self, shard: &mut Shard) {
        let sessions = self.layout();
        let mut event_index = self.plan.first_event;
        for session in &sessions {
            for &ts in session {
                self.event(shard, event_index, ts);
                event_index += 1;
            }
        }
        let starts: Vec<i64> = sessions.iter().map(|session| session[0]).collect();
        self.clicks(shard, &starts);
        self.links(shard);
    }

    fn alias(random: &mut Random, plan: &UserPlan, aliases: usize) -> usize {
        if aliases == 1 || (!plan.chain && random.chance(FIRST_ALIAS_SHARE)) {
            0
        } else {
            random.below(aliases as u64) as usize
        }
    }

    fn user_field(&mut self, id: u64) -> Option<u64> {
        if self.random.chance(self.dataset.rates.bad_user) {
            None
        } else if self.random.chance(self.dataset.rates.case_variant) {
            Some(uppercase_packed(id))
        } else {
            Some(id)
        }
    }

    fn layout(&mut self) -> Vec<Vec<i64>> {
        let random = &mut self.random;
        let count = self.plan.sessions as usize;
        let mut used = [false; SLOTS as usize];
        let mut slots: Vec<(usize, bool)> = Vec::with_capacity(count);
        if count == SLOTS as usize {
            slots.extend((0..count).map(|slot| (slot, false)));
        } else {
            let may_cross = count <= MOST_SESSIONS_TO_CROSS;
            for _ in 0..count {
                if may_cross && random.chance(MIDNIGHT_SHARE) {
                    let day = random.below(WINDOW_DAYS as u64 - 1) as usize;
                    let slot = day * SLOTS_PER_DAY + LAST_SLOT_OF_DAY;
                    if !used[slot] && !used[slot + 1] {
                        used[slot] = true;
                        used[slot + 1] = true;
                        slots.push((slot, true));
                        continue;
                    }
                }
                loop {
                    let slot = random.below(u64::from(SLOTS)) as usize;
                    if !used[slot] {
                        used[slot] = true;
                        slots.push((slot, false));
                        break;
                    }
                }
            }
            slots.sort_unstable();
        }

        let mut sizes = vec![1u64; count];
        let weights: Vec<u64> = (0..count).map(|_| 1 + random.below(4)).collect();
        let weight_total: u64 = weights.iter().sum();
        let extra = self.plan.events - count as u64;
        let mut given = 0;
        let mut running = 0;
        for (size, weight) in sizes.iter_mut().zip(&weights) {
            running += weight;
            let through = extra * running / weight_total;
            *size += through - given;
            given = through;
        }

        slots
            .iter()
            .zip(sizes)
            .map(|(&(slot, crossing), size)| {
                let mut gaps: Vec<i64> = (1..size).map(|_| random.spread(10) * SECOND).collect();
                let natural: i64 = gaps.iter().sum();
                if natural > LONGEST_SESSION {
                    for gap in &mut gaps {
                        *gap = *gap * LONGEST_SESSION / natural / SECOND * SECOND;
                    }
                }
                if let Some(at) = gaps.iter().position(|_| random.chance(BOUNDARY_GAP_SHARE)) {
                    gaps[at] = SESSION_GAP + random.between(0, 1);
                }
                let duration: i64 = gaps.iter().sum();
                let slot_start = self.dataset.window_start + slot as i64 * SLOT;
                let start = if crossing && duration >= 2 * SECOND {
                    slot_start + SLOT - random.between(1, duration / SECOND) * SECOND
                } else {
                    slot_start + random.below((LATEST_START / SECOND) as u64) as i64 * SECOND
                } + if random.chance(MILLISECOND_START_SHARE) {
                    random.between(1, 999)
                } else {
                    0
                };
                let mut ts = start;
                let mut times = Vec::with_capacity(size as usize);
                times.push(ts);
                for gap in gaps {
                    ts += gap;
                    times.push(ts);
                }
                times
            })
            .collect()
    }

    fn event(&mut self, shard: &mut Shard, index: u64, ts: i64) {
        let rates = &self.dataset.rates;
        let _event_id_case_variant = self.random.chance(rates.case_variant);
        let alias = Self::alias(&mut self.random, &self.plan, self.ids.len());
        let mut person = self.user_field(self.ids[alias]);

        let timestamp_is_bad = self.random.chance(rates.bad_timestamp);
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
        let ingest_ts = ingest(&mut self.random, rates.late, ts);

        let kind = self.random.below(100);
        let purchase = kind < 3;
        let type_valid = if self.random.chance(rates.bad_type) {
            let _ = self.random.below(8);
            false
        } else {
            true
        };
        let selected_product = self.dataset.product_by_rank[self
            .random
            .by_cumulative(&self.dataset.product_rank_cumulative)]
            as usize;
        let (product, product_present) = if self.random.chance(rates.unknown_product) {
            match self.random.below(10) {
                0 => (u32::MAX, false),
                1 => (u32::MAX, true),
                _ => {
                    let _ = self.random.between(PRODUCTS as i64 + 1, 99_999);
                    (u32::MAX, true)
                }
            }
        } else {
            (selected_product as u32, true)
        };
        let (price, price_valid) = if purchase {
            if self.random.chance(rates.bad_price) {
                let _ = self.random.below(9);
                (0, false)
            } else if self.random.chance(ODD_PRICE_SHARE) {
                let price = [10_000_000, 1, 250][self.random.below(3) as usize];
                (price, true)
            } else {
                (self.dataset.product_price[selected_product] as u32, true)
            }
        } else {
            let draw = self.random.below(20);
            if draw >= 19 {
                let _ = self.random.below(9);
            }
            (0, true)
        };
        let base_valid =
            person.is_some() && !timestamp_is_bad && type_valid && product_present && price_valid;

        let copies = if self.random.chance(rates.event_duplicates) {
            1 + self.random.below(3)
        } else {
            0
        };
        let special_bad_original =
            copies > 0 && !timestamp_is_bad && self.random.chance(BAD_ORIGINAL_SHARE);
        if special_bad_original {
            let outside = outside(
                &mut self.random,
                self.dataset.window_start,
                self.dataset.window_end(),
            );
            consume_bad_timestamp(&mut self.random, outside);
        }
        let original_present = emit(&mut self.random, rates, &self.dataset.event_parts, 7);
        let mut winner_recorded = original_present;
        if original_present && base_valid && !special_bad_original && ingest_ts - ts <= DAY {
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
            person = Some(user_id(self.dataset, other as u64 * ALIAS_SLOTS));
        }
        for _ in 0..copies {
            let later = ingest_ts + self.random.between(1, 600) * SECOND;
            let present = emit(&mut self.random, rates, &self.dataset.event_parts, 7);
            if present && !winner_recorded {
                winner_recorded = true;
                if base_valid && later - ts <= DAY {
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

    fn clicks(&mut self, shard: &mut Shard, session_starts: &[i64]) {
        let earliest = self.dataset.window_start - CLICK_REACH;
        let mut targets: Vec<i64> = session_starts
            .iter()
            .copied()
            .filter(|_| self.random.chance(TARGETED_SESSION_SHARE))
            .collect();
        if targets.is_empty() {
            targets.push(*self.random.pick(session_starts));
        }
        let mut index = self.plan.first_click;
        let end = index + self.plan.clicks;
        while index < end {
            let start = *self.random.pick(&targets);
            let delta = match self.random.below(1000) {
                0..4 => CLICK_REACH,
                4..8 => CLICK_REACH + 1,
                8..12 => 0,
                12..16 => -1,
                16..750 => (self.random.spread(19) * SECOND).min(CLICK_REACH),
                _ => CLICK_REACH + self.random.between(1, 3 * DAY / SECOND) * SECOND,
            };
            let mut ts = start - delta;
            if ts < earliest {
                ts = start
                    - self.random.below(((start - earliest) / SECOND) as u64 + 1) as i64 * SECOND;
            }
            let same_millisecond = end - index >= 2 && self.random.chance(SAME_MILLISECOND_SHARE);
            for _ in 0..1 + u64::from(same_millisecond) {
                self.click(shard, index, ts);
                index += 1;
            }
        }
    }

    fn click(&mut self, shard: &mut Shard, index: u64, ts: i64) {
        let rates = &self.dataset.rates;
        let id_case_variant = self.random.chance(rates.case_variant);
        let id = packed_id(
            b'k',
            scramble(index, self.dataset.click_key),
            id_case_variant,
        );
        let alias = Self::alias(&mut self.random, &self.plan, self.ids.len());
        let person = self.user_field(self.ids[alias]);
        let campaign = if self.random.chance(rates.unknown_campaign) {
            if self.random.chance(UPPERCASE_CAMPAIGN_SHARE) {
                let _ = self.random.between(1, CAMPAIGNS as i64);
            } else {
                let _ = self.random.between(CAMPAIGNS as i64 + 1, 9_999);
            }
            None
        } else if self.random.chance(WHALE_CAMPAIGN_SHARE) {
            Some(*self.random.pick(&self.dataset.whale_campaigns))
        } else {
            Some(
                self.dataset.campaign_by_rank[self
                    .random
                    .by_cumulative(&self.dataset.campaign_rank_cumulative)],
            )
        };
        let timestamp_is_bad = self.random.chance(rates.bad_timestamp);
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
        let copies = if self.random.chance(rates.click_duplicates) {
            1 + self.random.below(2)
        } else {
            0
        };
        let mut winner_recorded = false;
        for _ in 0..=copies {
            let present = emit(&mut self.random, rates, &self.dataset.click_parts, 4);
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
        let count = self.ids.len();
        let mut pairs = Vec::new();
        if self.plan.chain {
            let mut order: Vec<usize> = (0..count).collect();
            self.random.shuffle(&mut order);
            for pair in order.windows(2) {
                pairs.push((self.ids[pair[0]], self.ids[pair[1]]));
            }
        } else {
            for alias in 1..count {
                let earlier = self.random.below(alias as u64) as usize;
                pairs.push((self.ids[alias], self.ids[earlier]));
            }
        }
        if self.random.chance(UNUSED_ALIAS_SHARE) {
            let unused = user_id(self.dataset, self.index as u64 * ALIAS_SLOTS + count as u64);
            pairs.push((*self.random.pick(&self.ids), unused));
        }
        if self.plan.kiosk > 0 {
            let kiosk = user_id(self.dataset, KIOSK_BASE + u64::from(self.plan.kiosk));
            pairs.push((*self.random.pick(&self.ids), kiosk));
        }
        if self.random.chance(SELF_LINK_SHARE) {
            let id = *self.random.pick(&self.ids);
            pairs.push((id, id));
        }
        for (left, right) in pairs {
            let copies = 1 + usize::from(self.random.chance(REPEATED_LINK_SHARE));
            for _ in 0..copies {
                let (first, second) = if self.random.chance(0.5) {
                    (left, right)
                } else {
                    (right, left)
                };
                let first = self.link_field(first);
                let second = self.link_field(second);
                if emit(
                    &mut self.random,
                    &self.dataset.rates,
                    &self.dataset.link_parts,
                    2,
                ) {
                    if let (Some(first), Some(second)) = (first, second) {
                        shard.links.push((first, second));
                    }
                }
            }
        }
    }

    fn link_field(&mut self, id: u64) -> Option<u64> {
        if self.random.chance(self.dataset.rates.bad_user) {
            None
        } else if self.random.chance(LINK_CASE_VARIANT) {
            Some(uppercase_packed(id))
        } else {
            Some(id)
        }
    }
}

fn user_id(dataset: &Dataset, number: u64) -> u64 {
    packed_id(b'u', scramble(number, dataset.user_key), false)
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
    let draw = random.below(10);
    if milliseconds % SECOND == 0 && draw == 6 {
        let _ = random.below(31);
    }
}

fn consume_bad_timestamp(random: &mut Random, outside: i64) {
    match random.below(15) {
        7 => {
            let _ = random.between(1, 9);
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

fn ingest(random: &mut Random, late: f64, ts: i64) -> i64 {
    let floor = ts - ts.rem_euclid(SECOND);
    let ceiling = if floor == ts { ts } else { floor + SECOND };
    if random.chance(late) {
        match random.below(10) {
            0 if floor == ts => ts + DAY,
            1 if floor == ts => ts + DAY + SECOND,
            _ => ceiling + random.between(DAY / SECOND + 1, 5 * DAY / SECOND) * SECOND,
        }
    } else if random.chance(EARLY_INGEST_SHARE) {
        floor - random.between(1, 5) * SECOND
    } else {
        ceiling + random.between(0, 600) * SECOND
    }
}

fn emit(random: &mut Random, rates: &Rates, shares: &[f64], fields: usize) -> bool {
    for _ in 0..fields {
        let _ = random.chance(rates.quoted_field);
    }
    let mut present = true;
    if random.chance(rates.malformed) {
        let fault = random.below(6);
        present = fault == 5;
        if fault == 5 {
            let _ = random.by_cumulative(shares);
        }
    }
    let _ = random.by_cumulative(shares);
    present
}
