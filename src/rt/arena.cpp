#include <keel/rt/arena.hpp>

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace keel::rt {

BlockArena::BlockArena(int64_t capacity_bytes, int64_t block_size)
    : block_size_(block_size),
      backing_(nullptr, AlignedDeleter{static_cast<size_t>(
          block_size > 0 &&
                  static_cast<uint64_t>(block_size) <=
                      std::numeric_limits<size_t>::max()
              ? block_size
              : alignof(std::max_align_t))}) {
  if (block_size <= 0 || capacity_bytes <= 0)
    throw std::invalid_argument("arena capacity and block size must be positive");
  const auto is_power_of_two = [](int64_t value) {
    return (value & (value - 1)) == 0;
  };
  if (!is_power_of_two(block_size) ||
      block_size < static_cast<int64_t>(alignof(void*)) ||
      static_cast<uint64_t>(block_size) > std::numeric_limits<size_t>::max())
    throw std::invalid_argument(
        "block size must be a supported power-of-two alignment");
  if (capacity_bytes % block_size != 0)
    throw std::invalid_argument("capacity must be a multiple of block size");
  nblocks_ = capacity_bytes / block_size;
  if (nblocks_ <= 0 || nblocks_ > std::numeric_limits<int32_t>::max() ||
      static_cast<uint64_t>(capacity_bytes) > std::numeric_limits<size_t>::max())
    throw std::invalid_argument("arena size is not representable");
  capacity_bytes_ = capacity_bytes;
  backing_.reset(static_cast<std::byte*>(::operator new(
      static_cast<size_t>(capacity_bytes),
      std::align_val_t(static_cast<size_t>(block_size)))));
  owner_.assign((size_t)nblocks_, -1);
}

BlockArena::BlockArena(const BlockArena& other)
    : block_size_(other.block_size_),
      nblocks_(other.nblocks_),
      capacity_bytes_(other.capacity_bytes_),
      backing_(nullptr, AlignedDeleter{static_cast<size_t>(other.block_size_)}),
      owner_(), free_(), held_(), records_(), free_slots_() {
  std::lock_guard<std::recursive_mutex> lk(other.mu_);
  owner_ = other.owner_;
  free_ = other.free_;
  held_ = other.held_;
  records_ = other.records_;
  free_slots_ = other.free_slots_;
  backing_.reset(static_cast<std::byte*>(::operator new(
      static_cast<size_t>(capacity_bytes_),
      std::align_val_t(static_cast<size_t>(block_size_)))));
  std::memcpy(backing_.get(), other.backing_.get(),
              static_cast<size_t>(capacity_bytes_));
}

int64_t BlockArena::claim(TaskId t, int64_t nblocks) {
  std::lock_guard<std::recursive_mutex> lk(mu_);
  if (nblocks < 0 || nblocks > nblocks_)
    throw std::invalid_argument("claim size out of range");
  ensure_task(t);
  std::vector<int32_t> selected;
  selected.reserve(static_cast<size_t>(nblocks));
  for (int64_t i = 0; i < nblocks_ &&
                      static_cast<int64_t>(selected.size()) < nblocks;
       ++i) {
    if (owner_[(size_t)i] == -1) selected.push_back(static_cast<int32_t>(i));
  }
  if (static_cast<int64_t>(selected.size()) != nblocks)
    throw std::runtime_error("arena has insufficient unowned blocks");
  // Reserve before changing ownership. insert_free_run performs no allocation
  // after this reservation, so a vector allocation failure is still atomic.
  free_[t].reserve(free_[t].size() + selected.size());
  for (int32_t block : selected) {
    owner_[(size_t)block] = t;
    insert_free_run(t, block, 1);
  }
  return static_cast<int64_t>(selected.size());
}

int64_t BlockArena::blocks_owned(TaskId t) const {
  std::lock_guard<std::recursive_mutex> lk(mu_);
  int64_t n = 0;
  for (int64_t i = 0; i < nblocks_; ++i)
    if (owner_[(size_t)i] == t) ++n;
  return n;
}

int64_t BlockArena::bytes_owned(TaskId t) const {
  std::lock_guard<std::recursive_mutex> lk(mu_);
  int64_t b = blocks_owned(t), r;
  if (__builtin_mul_overflow(b, block_size_, &r))
    throw std::overflow_error("bytes_owned overflows");
  return r;
}

int64_t BlockArena::free_blocks(TaskId t) const {
  std::lock_guard<std::recursive_mutex> lk(mu_);
  int64_t total = 0;
  if (t < 0 || static_cast<size_t>(t) >= free_.size()) return 0;
  for (const FreeRun& run : free_[t]) {
    if (__builtin_add_overflow(total, static_cast<int64_t>(run.nblocks),
                               &total))
      throw std::overflow_error("free_blocks overflows");
  }
  return total;
}

int64_t BlockArena::largest_free_run_blocks(TaskId t) const {
  std::lock_guard<std::recursive_mutex> lk(mu_);
  if (t < 0 || static_cast<size_t>(t) >= free_.size()) return 0;
  int64_t largest = 0;
  for (const FreeRun& run : free_[t])
    if (run.nblocks > largest) largest = run.nblocks;
  return largest;
}

int64_t BlockArena::live_bytes(TaskId t) const {
  std::lock_guard<std::recursive_mutex> lk(mu_);
  int64_t sum = 0;
  for (const AllocRecord& r : records_) {
    if (!r.live || r.owner != t) continue;
    if (__builtin_add_overflow(sum, r.bytes, &sum))
      throw std::overflow_error("live_bytes overflows");
  }
  return sum;
}

void BlockArena::insert_free_run(TaskId t, int32_t first, int32_t n) {
  auto& runs = free_[t];
  // Insert sorted, then coalesce with neighbors.
  auto it = runs.begin();
  while (it != runs.end() && it->first < first) ++it;
  it = runs.insert(it, FreeRun{first, n});
  // coalesce next
  if (it + 1 != runs.end() && it->first + it->nblocks == (it + 1)->first) {
    it->nblocks += (it + 1)->nblocks;
    runs.erase(it + 1);
  }
  // coalesce prev
  if (it != runs.begin() && (it - 1)->first + (it - 1)->nblocks == first) {
    (it - 1)->nblocks += it->nblocks;
    runs.erase(it);
  }
}

AllocHandle BlockArena::alloc(TaskId t, int64_t bytes) {
  std::lock_guard<std::recursive_mutex> lk(mu_);
  ensure_task(t);
  if (bytes <= 0) throw std::invalid_argument("allocation size must be positive");
  // Overflow-safe ceil division: (bytes + block_size - 1) would overflow for
  // huge bytes (found by the adversarial suite), so divide first.
  int64_t need = bytes / block_size_ + (bytes % block_size_ != 0 ? 1 : 0);
  if (need > nblocks_) throw std::invalid_argument("allocation larger than arena");
  auto& runs = free_[t];
  for (size_t i = 0; i < runs.size(); ++i) {
    if (runs[i].nblocks >= need) {
      int32_t first = runs[i].first;
      uint32_t slot;
      if (!free_slots_.empty()) {
        slot = free_slots_.back();
      } else {
        if (records_.size() == std::numeric_limits<uint32_t>::max())
          throw std::overflow_error("allocation record slots exhausted");
        records_.reserve(records_.size() + 1);
        records_.emplace_back();
        slot = static_cast<uint32_t>(records_.size() - 1);
      }
      AllocRecord& r = records_[slot];
      if (r.generation == std::numeric_limits<uint64_t>::max())
        throw std::overflow_error("allocation generation exhausted");
      // shrink or remove the run
      if (runs[i].nblocks == need)
        runs.erase(runs.begin() + (long)i);
      else {
        runs[i].first += (int32_t)need;
        runs[i].nblocks -= (int32_t)need;
      }
      if (!free_slots_.empty()) {
        free_slots_.pop_back();
      }
      r.generation += 1;
      r.owner = t;
      r.first_block = first;
      r.nblocks = (int32_t)need;
      r.bytes = bytes;
      r.live = true;
      return AllocHandle{slot, r.generation};
    }
  }
  throw std::runtime_error("insufficient contiguous storage for task (fragmentation respected)");
}

void BlockArena::free(AllocHandle h) {
  std::lock_guard<std::recursive_mutex> lk(mu_);
  if (!h.valid() || h.slot >= records_.size())
    throw std::invalid_argument("free: invalid handle");
  AllocRecord& r = records_[h.slot];
  if (!r.live || r.generation != h.generation)
    throw std::invalid_argument("free: stale or dead handle");
  free_[r.owner].reserve(free_[r.owner].size() + 1);
  free_slots_.reserve(free_slots_.size() + 1);
  insert_free_run(r.owner, r.first_block, r.nblocks);
  r.live = false;
  free_slots_.push_back(h.slot);
}

AllocRecord BlockArena::hold_allocation(TaskId t, AllocHandle h) {
  std::lock_guard<std::recursive_mutex> lk(mu_);
  ensure_task(t);
  if (!h.valid() || h.slot >= records_.size())
    throw std::invalid_argument("hold_allocation: invalid handle");
  AllocRecord& r = records_[h.slot];
  if (!r.live || r.generation != h.generation || r.owner != t)
    throw std::invalid_argument("hold_allocation: stale, dead, or foreign handle");

  // Prepare every vector capacity before changing the record or held index.
  held_[t].reserve(held_[t].size() + 1);
  free_slots_.reserve(free_slots_.size() + 1);
  const AllocRecord result = r;
  held_[t].push_back({r.first_block, r.nblocks});
  std::sort(held_[t].begin(), held_[t].end(),
            [](const FreeRun& a, const FreeRun& b) { return a.first < b.first; });
  for (size_t i = 0; i + 1 < held_[t].size();) {
    if (held_[t][i].first + held_[t][i].nblocks == held_[t][i + 1].first) {
      held_[t][i].nblocks += held_[t][i + 1].nblocks;
      held_[t].erase(held_[t].begin() + static_cast<long>(i + 1));
    } else {
      ++i;
    }
  }
  r.live = false;
  free_slots_.push_back(h.slot);
  return result;
}

std::vector<int32_t> BlockArena::transfer_allocation(TaskId from, TaskId to,
                                                     AllocHandle h) {
  std::lock_guard<std::recursive_mutex> lk(mu_);
  if (from == to) throw std::invalid_argument("transfer_allocation: same owner");
  ensure_task(from);
  ensure_task(to);
  if (!h.valid() || h.slot >= records_.size())
    throw std::invalid_argument("transfer_allocation: invalid handle");
  AllocRecord& r = records_[h.slot];
  if (!r.live || r.generation != h.generation || r.owner != from)
    throw std::invalid_argument("transfer_allocation: stale, dead, or foreign handle");

  std::vector<int32_t> moved;
  moved.reserve(static_cast<size_t>(r.nblocks));
  for (int32_t b = r.first_block; b < r.first_block + r.nblocks; ++b)
    moved.push_back(b);
  free_[to].reserve(free_[to].size() + 1);
  free_slots_.reserve(free_slots_.size() + 1);
  const int32_t first = r.first_block;
  const int32_t n = r.nblocks;
  r.live = false;
  r.owner = to;
  free_slots_.push_back(h.slot);
  for (int32_t b : moved) owner_[static_cast<size_t>(b)] = to;
  insert_free_run(to, first, n);
  return moved;
}

void* BlockArena::ptr(AllocHandle h) {
  std::lock_guard<std::recursive_mutex> lk(mu_);
  const BlockArena* c = this;
  return const_cast<void*>(c->ptr(h));
}

const void* BlockArena::ptr(AllocHandle h) const {
  std::lock_guard<std::recursive_mutex> lk(mu_);
  const AllocRecord r = record(h);
  if (!r.live || r.generation != h.generation)
    throw std::invalid_argument("ptr: stale or dead handle");
  return backing_.get() + (size_t)r.first_block * (size_t)block_size_;
}

AllocRecord BlockArena::record(AllocHandle h) const {
  std::lock_guard<std::recursive_mutex> lk(mu_);
  if (!h.valid() || h.slot >= records_.size())
    throw std::invalid_argument("record: invalid handle");
  return records_[h.slot];
}

std::vector<int32_t> BlockArena::transfer_free(TaskId from, TaskId to,
                                               int64_t min_blocks) {
  std::lock_guard<std::recursive_mutex> lk(mu_);
  if (min_blocks <= 0) throw std::invalid_argument("transfer must move >0 blocks");
  ensure_task(from);
  ensure_task(to);
  auto& src = free_[from];
  // Atomicity: verify availability BEFORE moving anything (audit-style
  // physical atomicity — a failed transfer must leave the arena untouched).
  int64_t available = 0;
  for (const FreeRun& r : src)
    if (__builtin_add_overflow(available, static_cast<int64_t>(r.nblocks),
                               &available))
      throw std::overflow_error("transfer: block count overflows");
  if (available < min_blocks)
    throw std::runtime_error("transfer: source cannot supply enough free blocks");
  std::vector<size_t> order(src.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return src[a].nblocks > src[b].nblocks;
  });
  std::vector<size_t> chosen;
  int64_t chosen_blocks = 0;
  for (size_t idx : order) {
    if (chosen_blocks >= min_blocks) break;
    chosen.push_back(idx);
    if (__builtin_add_overflow(chosen_blocks,
                               static_cast<int64_t>(src[idx].nblocks),
                               &chosen_blocks))
      throw std::overflow_error("transfer: selected block count overflows");
  }
  std::vector<int32_t> moved;
  moved.reserve(static_cast<size_t>(chosen_blocks));
  free_[to].reserve(free_[to].size() + chosen.size());
  // All potentially-throwing work is complete. Move whole free runs, largest
  // first (fewer fragmented leftovers), in a non-throwing metadata phase.
  for (size_t idx : chosen) {
    FreeRun run = src[idx];
    src[idx].nblocks = 0;
    for (int32_t b = run.first; b < run.first + run.nblocks; ++b) {
      owner_[(size_t)b] = to;
      moved.push_back(b);
    }
    insert_free_run(to, run.first, run.nblocks);
  }
  src.erase(std::remove_if(src.begin(), src.end(),
                           [](const FreeRun& r) { return r.nblocks == 0; }),
            src.end());
  return moved;
}

void BlockArena::hold(TaskId t, int32_t first, int32_t n) {
  std::lock_guard<std::recursive_mutex> lk(mu_);
  ensure_task(t);
  if (first < 0 || n <= 0 || static_cast<int64_t>(first) + n > nblocks_)
    throw std::invalid_argument("hold: range out of bounds");
  auto& src = free_[t];
  for (size_t i = 0; i < src.size(); ++i) {
    const FreeRun& r = src[i];
    if (first >= r.first && static_cast<int64_t>(first) + n <=
                                  static_cast<int64_t>(r.first) + r.nblocks) {
      std::vector<FreeRun> replacement;
      if (first > r.first)
        replacement.push_back(FreeRun{r.first, first - r.first});
      if (first + n < r.first + r.nblocks)
        replacement.push_back(
            FreeRun{int32_t(first + n), int32_t(r.first + r.nblocks - first - n)});
      src.reserve(src.size() + replacement.size());
      auto& h = held_[t];
      h.reserve(h.size() + 1);
      src.erase(src.begin() + (long)i);
      for (auto it = replacement.rbegin(); it != replacement.rend(); ++it)
        src.insert(src.begin() + (long)i, *it);
      // Held runs coalesce like free runs.
      h.push_back(FreeRun{first, n});
      std::sort(h.begin(), h.end(),
                [](const FreeRun& a, const FreeRun& b) { return a.first < b.first; });
      for (size_t j = 0; j + 1 < h.size();) {
        if (h[j].first + h[j].nblocks == h[j + 1].first) {
          h[j].nblocks += h[j + 1].nblocks;
          h.erase(h.begin() + (long)(j + 1));
        } else {
          ++j;
        }
      }
      return;
    }
  }
  throw std::runtime_error("hold: range not free-owned by task");
}

void BlockArena::unhold(TaskId t, int32_t first, int32_t n) {
  std::lock_guard<std::recursive_mutex> lk(mu_);
  ensure_task(t);
  if (first < 0 || n <= 0 || static_cast<int64_t>(first) + n > nblocks_)
    throw std::invalid_argument("unhold: range out of bounds");
  auto& h = held_[t];
  for (size_t i = 0; i < h.size(); ++i) {
    const FreeRun& r = h[i];
    if (first >= r.first && static_cast<int64_t>(first) + n <=
                                  static_cast<int64_t>(r.first) + r.nblocks) {
      std::vector<FreeRun> replacement;
      if (first > r.first)
        replacement.push_back(FreeRun{r.first, first - r.first});
      if (first + n < r.first + r.nblocks)
        replacement.push_back(
            FreeRun{int32_t(first + n), int32_t(r.first + r.nblocks - first - n)});
      auto& dst = free_[t];
      h.reserve(h.size() + replacement.size());
      dst.reserve(dst.size() + 1);
      h.erase(h.begin() + (long)i);
      for (auto it = replacement.rbegin(); it != replacement.rend(); ++it)
        h.insert(h.begin() + (long)i, *it);
      insert_free_run(t, first, n);
      return;
    }
  }
  throw std::runtime_error("unhold: range not held by task");
}

std::vector<int32_t> BlockArena::transfer_extent(TaskId from, TaskId to,
                                                 int32_t first, int32_t n) {
  std::lock_guard<std::recursive_mutex> lk(mu_);
  if (first < 0 || n <= 0 || static_cast<int64_t>(first) + n > nblocks_)
    throw std::invalid_argument("transfer_extent: range out of bounds");
  if (from == to) throw std::invalid_argument("transfer_extent: same owner");
  ensure_task(from);
  ensure_task(to);
  auto& src = free_[from];
  for (size_t i = 0; i < src.size(); ++i) {
    const FreeRun& r = src[i];
    if (first >= r.first && static_cast<int64_t>(first) + n <=
                                  static_cast<int64_t>(r.first) + r.nblocks) {
      // Split the run around the moved extent.
      std::vector<FreeRun> replacement;
      if (first > r.first) replacement.push_back(FreeRun{r.first, first - r.first});
      if (first + n < r.first + r.nblocks)
        replacement.push_back(
            FreeRun{int32_t(first + n), int32_t(r.first + r.nblocks - first - n)});
      std::vector<int32_t> moved;
      moved.reserve(static_cast<size_t>(n));
      src.reserve(src.size() + replacement.size());
      free_[to].reserve(free_[to].size() + 1);
      src.erase(src.begin() + (long)i);
      for (auto it = replacement.rbegin(); it != replacement.rend(); ++it)
        src.insert(src.begin() + (long)i, *it);
      for (int32_t b = first; b < first + n; ++b) {
        owner_[(size_t)b] = to;
        moved.push_back(b);
      }
      insert_free_run(to, first, n);
      return moved;
    }
  }
  // Held ranges (handover staging) are also transferable by their owner.
  if ((size_t)from < held_.size()) {
    auto& h = held_[from];
    for (size_t i = 0; i < h.size(); ++i) {
      const FreeRun& r = h[i];
      if (first >= r.first && static_cast<int64_t>(first) + n <=
                                    static_cast<int64_t>(r.first) + r.nblocks) {
        std::vector<FreeRun> replacement;
        if (first > r.first)
          replacement.push_back(FreeRun{r.first, first - r.first});
        if (first + n < r.first + r.nblocks)
          replacement.push_back(FreeRun{int32_t(first + n),
                                        int32_t(r.first + r.nblocks - first - n)});
        std::vector<int32_t> moved;
        moved.reserve(static_cast<size_t>(n));
        h.reserve(h.size() + replacement.size());
        free_[to].reserve(free_[to].size() + 1);
        h.erase(h.begin() + (long)i);
        for (auto it = replacement.rbegin(); it != replacement.rend(); ++it)
          h.insert(h.begin() + (long)i, *it);
        for (int32_t b = first; b < first + n; ++b) {
          owner_[(size_t)b] = to;
          moved.push_back(b);
        }
        insert_free_run(to, first, n);
        return moved;
      }
    }
  }
  throw std::runtime_error("transfer_extent: range not free-owned by source");
}

std::vector<int32_t> BlockArena::transfer_held_extents(
    TaskId from, TaskId to,
    const std::vector<std::pair<int32_t, int32_t>>& extents) {
  std::lock_guard<std::recursive_mutex> lk(mu_);
  if (from == to) throw std::invalid_argument("transfer_held_extents: same owner");
  if (extents.empty())
    throw std::invalid_argument("transfer_held_extents: empty transaction");
  ensure_task(from);
  ensure_task(to);

  std::vector<std::pair<int32_t, int32_t>> ordered = extents;
  std::sort(ordered.begin(), ordered.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  int64_t total = 0;
  int32_t previous_end = -1;
  auto contains = [](const std::vector<FreeRun>& runs, int32_t first,
                     int32_t n) {
    for (const FreeRun& r : runs) {
      if (first >= r.first && static_cast<int64_t>(first) + n <=
                                    static_cast<int64_t>(r.first) + r.nblocks)
        return true;
    }
    return false;
  };
  for (const auto& e : ordered) {
    const int32_t first = e.first;
    const int32_t n = e.second;
    if (first < 0 || n <= 0 || static_cast<int64_t>(first) + n > nblocks_ ||
        (previous_end >= 0 && first < previous_end) ||
        !contains(held_[from], first, n))
      throw std::runtime_error(
          "transfer_held_extents: extent is not a disjoint held range");
    previous_end = static_cast<int32_t>(static_cast<int64_t>(first) + n);
    if (__builtin_add_overflow(total, static_cast<int64_t>(n), &total))
      throw std::overflow_error("transfer_held_extents: size overflow");
  }

  // Build the post-transaction indexes completely before changing owner_.
  // Vector move-assignment and the owner bitmap writes below are non-throwing.
  std::vector<FreeRun> new_held = held_[from];
  std::vector<FreeRun> new_free = free_[to];
  new_held.reserve(new_held.size() + ordered.size());
  new_free.reserve(new_free.size() + ordered.size());
  std::vector<int32_t> moved;
  moved.reserve(static_cast<size_t>(total));

  auto remove_range = [](std::vector<FreeRun>& runs, int32_t first,
                         int32_t n) {
    for (size_t i = 0; i < runs.size(); ++i) {
      FreeRun r = runs[i];
      if (first < r.first || static_cast<int64_t>(first) + n >
                                  static_cast<int64_t>(r.first) + r.nblocks)
        continue;
      std::vector<FreeRun> replacement;
      if (first > r.first) replacement.push_back({r.first, first - r.first});
      const int64_t end = static_cast<int64_t>(first) + n;
      const int64_t run_end = static_cast<int64_t>(r.first) + r.nblocks;
      if (end < run_end)
        replacement.push_back(
            {static_cast<int32_t>(end), static_cast<int32_t>(run_end - end)});
      runs.erase(runs.begin() + static_cast<long>(i));
      for (auto it = replacement.rbegin(); it != replacement.rend(); ++it)
        runs.insert(runs.begin() + static_cast<long>(i), *it);
      return;
    }
    throw std::logic_error("range disappeared during arena transaction");
  };

  for (const auto& e : ordered) {
    remove_range(new_held, e.first, e.second);
    for (int32_t b = e.first; b < e.first + e.second; ++b)
      moved.push_back(b);
    new_free.push_back({e.first, e.second});
  }
  std::sort(new_free.begin(), new_free.end(),
            [](const FreeRun& a, const FreeRun& b) { return a.first < b.first; });
  std::vector<FreeRun> coalesced;
  coalesced.reserve(new_free.size());
  for (const FreeRun& r : new_free) {
    if (!coalesced.empty() &&
        coalesced.back().first + coalesced.back().nblocks == r.first)
      coalesced.back().nblocks += r.nblocks;
    else
      coalesced.push_back(r);
  }

  for (int32_t b : moved) owner_[(size_t)b] = to;
  held_[from] = std::move(new_held);
  free_[to] = std::move(coalesced);
  return moved;
}

void BlockArena::unhold_extents(
    TaskId t, const std::vector<std::pair<int32_t, int32_t>>& extents) {
  std::lock_guard<std::recursive_mutex> lk(mu_);
  if (extents.empty()) return;
  ensure_task(t);
  std::vector<std::pair<int32_t, int32_t>> ordered = extents;
  std::sort(ordered.begin(), ordered.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  std::vector<FreeRun> new_held = held_[t];
  std::vector<FreeRun> new_free = free_[t];
  new_held.reserve(new_held.size() + ordered.size());
  new_free.reserve(new_free.size() + ordered.size());
  int32_t previous_end = -1;
  auto remove_range = [](std::vector<FreeRun>& runs, int32_t first,
                         int32_t n) {
    for (size_t i = 0; i < runs.size(); ++i) {
      FreeRun r = runs[i];
      if (first < r.first || static_cast<int64_t>(first) + n >
                                  static_cast<int64_t>(r.first) + r.nblocks)
        continue;
      std::vector<FreeRun> replacement;
      if (first > r.first)
        replacement.push_back({r.first, first - r.first});
      const int64_t end = static_cast<int64_t>(first) + n;
      const int64_t run_end = static_cast<int64_t>(r.first) + r.nblocks;
      if (end < run_end)
        replacement.push_back({static_cast<int32_t>(end),
                               static_cast<int32_t>(run_end - end)});
      runs.erase(runs.begin() + static_cast<long>(i));
      for (auto it = replacement.rbegin(); it != replacement.rend(); ++it)
        runs.insert(runs.begin() + static_cast<long>(i), *it);
      return;
    }
    throw std::logic_error("range disappeared during unhold transaction");
  };
  for (const auto& e : ordered) {
    if (e.first < 0 || e.second <= 0 ||
        static_cast<int64_t>(e.first) + e.second > nblocks_ ||
        (previous_end >= 0 && e.first < previous_end))
      throw std::invalid_argument("unhold_extents: overlapping/out-of-bounds range");
    remove_range(new_held, e.first, e.second);
    new_free.push_back({e.first, e.second});
    previous_end = static_cast<int32_t>(static_cast<int64_t>(e.first) + e.second);
  }
  std::sort(new_free.begin(), new_free.end(),
            [](const FreeRun& a, const FreeRun& b) { return a.first < b.first; });
  std::vector<FreeRun> coalesced;
  coalesced.reserve(new_free.size());
  for (const FreeRun& r : new_free) {
    if (!coalesced.empty() &&
        coalesced.back().first + coalesced.back().nblocks == r.first)
      coalesced.back().nblocks += r.nblocks;
    else
      coalesced.push_back(r);
  }
  held_[t] = std::move(new_held);
  free_[t] = std::move(coalesced);
}

bool BlockArena::verify(std::string* err) const {
  std::lock_guard<std::recursive_mutex> lk(mu_);
  auto fail = [&](const std::string& m) {
    if (err) *err = m;
    return false;
  };
  // 1. Block ownership map vs per-task free-run tables.
  std::vector<char> in_free_run((size_t)nblocks_, 0);
  for (size_t t = 0; t < free_.size(); ++t) {
    if ((int64_t)t >= nblocks_ + 1 && free_[t].empty()) continue;
    int64_t prev_end = -1;
    for (const FreeRun& r : free_[t]) {
      if (r.nblocks <= 0) return fail("free run with non-positive length");
      if (r.first < 0 || r.first + r.nblocks > nblocks_)
        return fail("free run out of arena bounds");
      if (r.first <= prev_end) return fail("free runs overlap or unsorted");
      prev_end = r.first + r.nblocks - 1;
      for (int32_t b = r.first; b < r.first + r.nblocks; ++b) {
        if (owner_[(size_t)b] != (TaskId)t)
          return fail("free-run table disagrees with block ownership");
        if (in_free_run[(size_t)b]) return fail("block in two free runs");
        in_free_run[(size_t)b] = 1;
      }
    }
  }
  // 2. Live records: within their task's ownership, no overlap with free runs
  //    or with each other.
  std::vector<char> in_live((size_t)nblocks_, 0);
  int64_t live_total = 0;
  for (const AllocRecord& r : records_) {
    if (!r.live) continue;
    if (r.first_block < 0 || r.first_block + r.nblocks > nblocks_)
      return fail("live record out of arena bounds");
    if (r.bytes <= 0 || r.bytes > (int64_t)r.nblocks * block_size_)
      return fail("live record byte count inconsistent");
    for (int32_t b = r.first_block; b < r.first_block + r.nblocks; ++b) {
      if (owner_[(size_t)b] != r.owner)
        return fail("live record block not owned by its task");
      if (in_free_run[(size_t)b]) return fail("live record overlaps free run");
      if (in_live[(size_t)b]) return fail("two live records share a block");
      in_live[(size_t)b] = 1;
    }
    if (__builtin_add_overflow(live_total, static_cast<int64_t>(r.nblocks),
                               &live_total))
      return fail("live block count overflows");
  }
  // Held runs: owned by the task, not allocatable, not live.
  std::vector<char> in_held((size_t)nblocks_, 0);
  for (size_t t = 0; t < held_.size(); ++t) {
    for (const FreeRun& r : held_[t]) {
      if (r.nblocks <= 0) return fail("held run with non-positive length");
      if (r.first < 0 || r.first + r.nblocks > nblocks_)
        return fail("held run out of arena bounds");
      for (int32_t b = r.first; b < r.first + r.nblocks; ++b) {
        if (owner_[(size_t)b] != (TaskId)t)
          return fail("held-run table disagrees with block ownership");
        if (in_held[(size_t)b]) return fail("block in two held runs");
        if (in_free_run[(size_t)b]) return fail("block free and held");
        if (in_live[(size_t)b]) return fail("block live and held");
        in_held[(size_t)b] = 1;
      }
    }
  }
  // 3. Every owned block is live, free, or held for its owner.
  for (int64_t b = 0; b < nblocks_; ++b) {
    TaskId o = owner_[(size_t)b];
    if (o == -1) {
      if (in_live[(size_t)b]) return fail("unowned block holds a live record");
      continue;
    }
    if (o >= (int64_t)free_.size()) return fail("owner id out of range");
    if (!in_free_run[(size_t)b] && !in_live[(size_t)b] &&
        !in_held[(size_t)b])
      return fail("owned block is neither live nor free nor held");
  }
  if (live_total > nblocks_) return fail("live blocks exceed arena");
  return true;
}

}  // namespace keel::rt
