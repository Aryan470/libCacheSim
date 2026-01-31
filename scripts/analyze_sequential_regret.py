#!/usr/bin/env python3
"""
Analyze sequential regret in LRU cache eviction.

Sequential regret: when we miss on an object whose key is sequential (±10)
to a recently evicted object's key.
"""

import csv
import argparse
from collections import OrderedDict
from tqdm import tqdm


class LRUCache:
    """Simple LRU cache simulation."""

    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self.cache = OrderedDict()  # key -> (size, key_prefix)

    def access(self, key, size, key_prefix):
        """Returns (hit, evicted_keys_with_prefixes)"""
        evicted = []

        if key in self.cache:
            # Hit - move to end (MRU)
            self.cache.move_to_end(key)
            return True, evicted

        # Miss - need to insert
        # First evict until we have space
        while self.current_bytes + size > self.max_bytes and self.cache:
            evicted_key, (evicted_size, evicted_prefix) = self.cache.popitem(last=False)
            self.current_bytes -= evicted_size
            evicted.append((evicted_key, evicted_prefix))

        # Insert if it fits
        if size <= self.max_bytes:
            self.cache[key] = (size, key_prefix)
            self.current_bytes += size

        return False, evicted


class RegretTracker:
    """Track recent evictions for regret analysis."""

    def __init__(self, buffer_size=10000):
        self.buffer_size = buffer_size
        self.recent_evictions = {}  # key -> key_prefix
        self.eviction_order = []

    def add_eviction(self, key, key_prefix):
        if len(self.eviction_order) >= self.buffer_size:
            old_key = self.eviction_order.pop(0)
            if old_key in self.recent_evictions:
                del self.recent_evictions[old_key]
        self.recent_evictions[key] = key_prefix
        self.eviction_order.append(key)

    def was_recently_evicted(self, key):
        return key in self.recent_evictions

    def has_sequential_eviction(self, key_prefix):
        """Check if any recently evicted object has sequential key (±10)."""
        try:
            p = int(key_prefix)
        except:
            return False

        for evicted_prefix in self.recent_evictions.values():
            try:
                ep = int(evicted_prefix)
                if abs(p - ep) == 10:
                    return True
            except:
                continue
        return False


def run_simulation(trace_path, cache_size_gb, num_requests, regret_buffer_size):
    cache = LRUCache(int(cache_size_gb * 1024**3))
    tracker = RegretTracker(regret_buffer_size)

    n_requests = 0
    n_misses = 0
    n_regret = 0
    n_sequential_regret = 0

    with open(trace_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for row in tqdm(reader, total=num_requests, desc="Simulating LRU"):
            if n_requests >= num_requests:
                break

            key = row[1]  # full cacheKey
            size = int(row[3])  # objectSize
            key_prefix = key[:12]  # sequential part

            hit, evicted = cache.access(key, size, key_prefix)
            n_requests += 1

            if not hit:
                n_misses += 1

                # Check if this miss was on a recently evicted object
                if tracker.was_recently_evicted(key):
                    n_regret += 1

                # Check if this miss is sequential to any recently evicted object
                if tracker.has_sequential_eviction(key_prefix):
                    n_sequential_regret += 1

            # Record evictions
            for evicted_key, evicted_prefix in evicted:
                tracker.add_eviction(evicted_key, evicted_prefix)

    return {
        'n_requests': n_requests,
        'n_misses': n_misses,
        'n_regret': n_regret,
        'n_sequential_regret': n_sequential_regret,
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze sequential regret in LRU cache')
    parser.add_argument('trace', help='Path to CSV trace file')
    parser.add_argument('--cache-gb', type=float, default=1.0, help='Cache size in GB (default: 1)')
    parser.add_argument('--num-requests', type=int, default=5_000_000, help='Number of requests to process')
    parser.add_argument('--regret-buffer', type=int, default=10000, help='Regret buffer size')
    args = parser.parse_args()

    print(f"Cache size: {args.cache_gb} GB")
    print(f"Requests: {args.num_requests:,}")
    print(f"Regret buffer: {args.regret_buffer:,}")
    print()

    results = run_simulation(args.trace, args.cache_gb, args.num_requests, args.regret_buffer)

    n_requests = results['n_requests']
    n_misses = results['n_misses']
    n_regret = results['n_regret']
    n_sequential_regret = results['n_sequential_regret']

    print(f"\n=== Results ===")
    print(f"Total requests: {n_requests:,}")
    print(f"Total misses: {n_misses:,} ({100*n_misses/n_requests:.2f}%)")
    print(f"Regret misses (same object): {n_regret:,} ({100*n_regret/n_misses:.2f}% of misses)")
    print(f"Sequential regret (key ±10): {n_sequential_regret:,} ({100*n_sequential_regret/n_misses:.2f}% of misses)")


if __name__ == '__main__':
    main()
