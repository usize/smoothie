use std::sync::atomic::{AtomicU32, Ordering};

/// Adaptive semaphore whose ceiling can change at runtime.
///
/// Uses a CAS loop for lock-free `try_acquire`. When the AIMD controller
/// reduces the ceiling below the current `active` count, existing streams
/// are NOT evicted — they complete normally. Only new admissions are blocked.
pub struct AdaptiveSemaphore {
    active: AtomicU32,
}

impl AdaptiveSemaphore {
    /// Create a new semaphore with zero active streams.
    pub fn new() -> Self {
        Self {
            active: AtomicU32::new(0),
        }
    }

    /// Try to acquire a slot. Returns `true` if admitted, `false` if at ceiling.
    ///
    /// Uses a compare-and-swap loop: if `active < ceiling`, atomically
    /// increment and return true. Otherwise return false without blocking.
    pub fn try_acquire(&self, ceiling: u32) -> bool {
        loop {
            let current = self.active.load(Ordering::Acquire);
            if current >= ceiling {
                return false;
            }
            match self.active.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(_) => continue, // Retry on contention.
            }
        }
    }

    /// Release a slot when a stream completes.
    pub fn release(&self) {
        self.active.fetch_sub(1, Ordering::Release);
    }

    /// Current number of active streams (used for observability and tests).
    #[allow(dead_code)]
    pub fn active(&self) -> u32 {
        self.active.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquire_within_ceiling() {
        let sem = AdaptiveSemaphore::new();
        assert!(sem.try_acquire(3));
        assert!(sem.try_acquire(3));
        assert!(sem.try_acquire(3));
        assert!(!sem.try_acquire(3), "should reject at ceiling");
        assert_eq!(sem.active(), 3);
    }

    #[test]
    fn release_frees_slot() {
        let sem = AdaptiveSemaphore::new();
        assert!(sem.try_acquire(1));
        assert!(!sem.try_acquire(1));
        sem.release();
        assert_eq!(sem.active(), 0);
        assert!(sem.try_acquire(1), "should succeed after release");
    }

    #[test]
    fn ceiling_zero_rejects_all() {
        let sem = AdaptiveSemaphore::new();
        assert!(!sem.try_acquire(0));
    }

    #[test]
    fn dynamic_ceiling_reduction() {
        let sem = AdaptiveSemaphore::new();
        // Acquire 3 slots with ceiling=5.
        assert!(sem.try_acquire(5));
        assert!(sem.try_acquire(5));
        assert!(sem.try_acquire(5));

        // Reduce ceiling to 2: existing 3 streams stay, but new ones are blocked.
        assert!(!sem.try_acquire(2));
        assert_eq!(sem.active(), 3);

        // Release one, still at 2 which equals the new ceiling.
        sem.release();
        assert!(!sem.try_acquire(2), "still at ceiling after one release");

        sem.release();
        assert!(sem.try_acquire(2), "should succeed after dropping below ceiling");
    }

    #[test]
    fn concurrent_acquire_release() {
        use std::sync::Arc;
        use std::thread;

        let sem = Arc::new(AdaptiveSemaphore::new());
        let mut handles = Vec::new();

        // Spawn 100 threads that each acquire and release.
        for _ in 0..100 {
            let sem = Arc::clone(&sem);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    // Try acquire with high ceiling so all succeed.
                    if sem.try_acquire(200) {
                        sem.release();
                    }
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(sem.active(), 0, "all slots should be released after contention test");
    }
}
