/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Search-time breakdown profiler (Compute / Enqueue / Other).
//!
//! Compiled in only with `RUSTFLAGS="--cfg diskann_profile"`; otherwise [`now`] returns 0
//! and [`add`] is a no-op, so the normal build is not perturbed. When compiled in, timing
//! can also be switched off at runtime ([`set_enabled`]) so one binary measures a clean
//! pass and a profiled pass. Counters are process-wide with relaxed load/store (no locked
//! RMW), so profile single-threaded searches only.
//!
//! Timestamps are `lfence; rdtsc`: the fence makes the timestamp wait for all earlier
//! instructions (including cache-missing loads), so a stall is charged to the region that
//! caused it instead of drifting into a later one. Fencing removes some out-of-order
//! overlap, so report the breakdown scaled to the clean-pass latency.
//!
//! Phase -> category:
//! * Compute: [`Phase::DistComp`] (distance evaluations in `expand_beam` and start points)
//! * Enqueue: [`Phase::QueueMgmt`] (`closest_notvisited` pops + `best.insert`)
//! * Other:   [`Phase::Total`] minus the above (neighbor-list fetch + visited filter
//!   ([`Phase::NbrFilter`]), post-processing, loop overhead)

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Profiled phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Distance computations.
    DistComp = 0,
    /// Neighbor-list fetch + visited-set filtering inside `expand_beam`.
    NbrFilter,
    /// Candidate queue pops and inserts.
    QueueMgmt,
    /// Whole per-query search call (including post-processing).
    Total,
}

/// Number of phases.
pub const NUM_PHASES: usize = 4;

/// Extra counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Counter {
    /// Number of profiled queries.
    Queries = 0,
    /// Distance computations.
    DistComps,
    /// Expanded vertices.
    Hops,
}

const NUM_COUNTERS: usize = 3;

#[allow(clippy::declare_interior_mutable_const)]
const ZERO: AtomicU64 = AtomicU64::new(0);
static TICKS: [AtomicU64; NUM_PHASES] = [ZERO; NUM_PHASES];
static COUNTS: [AtomicU64; NUM_COUNTERS] = [ZERO; NUM_COUNTERS];

/// Whether profiling is compiled in.
pub const ENABLED: bool = cfg!(diskann_profile);

static ON: AtomicBool = AtomicBool::new(true);

/// Switch timing on or off at runtime (no effect unless compiled in).
pub fn set_enabled(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

#[inline(always)]
fn active() -> bool {
    ENABLED && ON.load(Ordering::Relaxed)
}

/// Current ordered TSC value (0 when profiling is disabled).
#[inline(always)]
pub fn now() -> u64 {
    #[cfg(all(diskann_profile, target_arch = "x86_64"))]
    {
        if active() {
            // SAFETY: `lfence` and `rdtsc` are available on all x86_64 CPUs.
            unsafe {
                core::arch::x86_64::_mm_lfence();
                core::arch::x86_64::_rdtsc()
            }
        } else {
            0
        }
    }
    #[cfg(not(all(diskann_profile, target_arch = "x86_64")))]
    {
        0
    }
}

#[inline(always)]
fn bump(slot: &AtomicU64, n: u64) {
    if active() {
        slot.store(slot.load(Ordering::Relaxed).wrapping_add(n), Ordering::Relaxed);
    }
}

/// Charge `ticks` to `phase`.
#[inline(always)]
pub fn add(phase: Phase, ticks: u64) {
    bump(&TICKS[phase as usize], ticks);
}

/// Charge the time since `*t` to `phase` and restart `*t`.
#[inline(always)]
pub fn lap(phase: Phase, t: &mut u64) {
    if active() {
        let n = now();
        add(phase, n.wrapping_sub(*t));
        *t = n;
    }
}

/// Increment a counter.
#[inline(always)]
pub fn count(counter: Counter, n: u64) {
    bump(&COUNTS[counter as usize], n);
}

/// Snapshot of the accumulated counters.
#[derive(Debug, Clone, Copy, Default)]
pub struct Snapshot {
    /// Ticks per phase, indexed by `Phase as usize`.
    pub ticks: [u64; NUM_PHASES],
    /// Number of profiled queries.
    pub queries: u64,
    /// Distance computations.
    pub dist_comps: u64,
    /// Expanded vertices.
    pub hops: u64,
}

/// Current ticks of `phase` (used to subtract nested phases).
#[inline(always)]
pub fn get(phase: Phase) -> u64 {
    if active() {
        TICKS[phase as usize].load(Ordering::Relaxed)
    } else {
        0
    }
}

/// Read the counters and reset them to zero.
pub fn take() -> Snapshot {
    let mut s = Snapshot::default();
    for (dst, src) in s.ticks.iter_mut().zip(TICKS.iter()) {
        *dst = src.swap(0, Ordering::Relaxed);
    }
    s.queries = COUNTS[Counter::Queries as usize].swap(0, Ordering::Relaxed);
    s.dist_comps = COUNTS[Counter::DistComps as usize].swap(0, Ordering::Relaxed);
    s.hops = COUNTS[Counter::Hops as usize].swap(0, Ordering::Relaxed);
    s
}

/// TSC ticks per second, calibrated once against the system clock.
pub fn ticks_per_sec() -> f64 {
    static TPS: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *TPS.get_or_init(|| {
        let c0 = std::time::Instant::now();
        let t0 = raw_tsc();
        while c0.elapsed() < std::time::Duration::from_millis(200) {}
        let t1 = raw_tsc();
        (t1.wrapping_sub(t0)) as f64 / c0.elapsed().as_secs_f64()
    })
}

fn raw_tsc() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: `rdtsc` is available on all x86_64 CPUs.
        unsafe { core::arch::x86_64::_rdtsc() }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        0
    }
}
