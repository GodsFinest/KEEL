// KEEL Phase 4A — Real controlled memory arena.
//
// Physical model (docs/PHASE1_AUDIT_REMEDIATION.md; SPEC v1.1):
//  - One contiguous backing buffer of `capacity_bytes`, divided into fixed
//    blocks (default 256 B, which also provides the alignment guarantee:
//    every allocation offset is block-aligned).
//  - Each block is owned by exactly one task or is unowned (the transfer
//    pool). Task grants are block counts claimed from the arena.
//  - Allocations are contiguous block runs handed out first-fit from the
//    task's owned blocks; a live run never overlaps another.
//  - free() returns the run to the task's free segments (coalesced);
//    transfer_free() physically re-assigns free owned blocks to another
//    task, whose later allocations then provably reuse that exact storage.
//  - Allocation handles are (slot, generation) pairs; stale handles are
//    rejected, never misinterpreted.
//
// Concurrency contract: arena METADATA is control-plane-only. The runtime's
// scheduler/control thread performs all alloc/free/transfer calls; worker
// threads only read/write memory contents through pointers they were given
// and never touch arena metadata. This centralized-controller simplification
// is deliberate (mandate Part I: acceptable metadata synchronization).

#pragma once

#include <cstdint>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace keel::rt {

using TaskId = int;

struct AllocHandle {
  uint32_t slot = UINT32_MAX;
  uint64_t generation = 0;
  bool valid() const { return slot != UINT32_MAX; }
};

inline bool operator==(const AllocHandle& a, const AllocHandle& b) {
  return a.slot == b.slot && a.generation == b.generation;
}

struct AllocRecord {
  uint64_t generation = 0;
  TaskId owner = -1;
  int32_t first_block = -1;
  int32_t nblocks = 0;
  int64_t bytes = 0;  // requested bytes (<= nblocks * block_size)
  bool live = false;
};

class BlockArena {
 public:
  BlockArena(int64_t capacity_bytes, int64_t block_size = 256);
  BlockArena(const BlockArena& other);
  BlockArena& operator=(const BlockArena&) = delete;

  int64_t block_size() const { return block_size_; }
  int64_t nblocks() const { return nblocks_; }
  int64_t capacity_bytes() const { return capacity_bytes_; }

  // --- ownership -----------------------------------------------------------
  // Assign the lowest-indexed unowned blocks to `t` (its initial region).
  // Returns blocks assigned. Blocks already owned are never reassigned.
  int64_t claim(TaskId t, int64_t nblocks);
  // Release a task's entire ownership after all live allocations and held
  // extents have been retired. This makes arena lifetime independent of the
  // TaskRuntime object that temporarily controlled the grant.
  void release_task(TaskId t);
  int64_t blocks_owned(TaskId t) const;
  int64_t bytes_owned(TaskId t) const;
  // Allocator-shape diagnostics used by the evaluation harness. Held blocks
  // are intentionally excluded from free_blocks: they are not allocatable.
  int64_t free_blocks(TaskId t) const;
  int64_t largest_free_run_blocks(TaskId t) const;

  // --- allocation ----------------------------------------------------------
  // First-fit contiguous run from `t`'s owned free blocks. Throws on invalid
  // size or insufficient contiguous storage (fragmentation is respected:
  // enough total bytes does NOT imply an allocatable run).
  AllocHandle alloc(TaskId t, int64_t bytes);
  void free(AllocHandle h);
  // Atomically retire a live allocation into the owner's held inventory.
  // Used by RELEASE staging so the allocation cannot be freed and then lose
  // its held designation if a later metadata operation fails.
  AllocRecord hold_allocation(TaskId t, AllocHandle h);
  // Atomically retire a live allocation and transfer its exact physical
  // blocks to `to` as free-owned storage.
  std::vector<int32_t> transfer_allocation(TaskId from, TaskId to,
                                           AllocHandle h);
  void* ptr(AllocHandle h);              // live handles only
  const void* ptr(AllocHandle h) const;
  AllocRecord record(AllocHandle h) const;
  int64_t live_bytes(TaskId t) const;    // Σ live allocation bytes

  // --- physical transfer ---------------------------------------------------
  // Move `min_blocks` (or more, in whole free runs) of `from`'s FREE owned
  // blocks to `to`. Returns the exact block ids moved — the observable
  // physical transfer. Throws if `from` cannot supply enough.
  std::vector<int32_t> transfer_free(TaskId from, TaskId to,
                                     int64_t min_blocks);

  // Hold: remove a free run from the task's allocatable pool without a
  // record (used by the handover to guarantee staged extents cannot be
  // re-allocated between DROP and COMMIT). unhold returns them to the pool.
  void hold(TaskId t, int32_t first, int32_t n);
  void unhold(TaskId t, int32_t first, int32_t n);

  // Move EXACTLY the block range [first, first+n) from `from` to `to`. The
  // range must be entirely free-owned by `from` (it may be a slice of a
  // larger free run; the run is split). This is the primitive that lets a
  // task surrender exactly one dropped tensor's storage even after
  // coalescing merged it with neighbors. Returns the moved block ids.
  std::vector<int32_t> transfer_extent(TaskId from, TaskId to, int32_t first,
                                       int32_t n);

  // Transactional handover primitives. Every extent must be wholly held by
  // `from`; all validation and allocations needed by the metadata mutation
  // happen before any ownership bit changes. The returned block ids are the
  // exact physical transfer.
  std::vector<int32_t> transfer_held_extents(
      TaskId from, TaskId to,
      const std::vector<std::pair<int32_t, int32_t>>& extents);
  void unhold_extents(
      TaskId t, const std::vector<std::pair<int32_t, int32_t>>& extents);

  // --- debugging -----------------------------------------------------------
  // Full invariant scan: one owner per block, live records within their
  // task's ownership, no overlap, free-run tables consistent with the block
  // map, totals within capacity. Returns false and fills `err` on violation.
  bool verify(std::string* err) const;

 private:
  struct FreeRun {
    int32_t first;
    int32_t nblocks;
  };
  void insert_free_run(TaskId t, int32_t first, int32_t n);  // coalescing
  std::vector<FreeRun>& free_runs(TaskId t) { return free_[t]; }
  const std::vector<FreeRun>& free_runs(TaskId t) const { return free_[t]; }
  void ensure_task(TaskId t) {
    if (t < 0) throw std::invalid_argument("negative task id");
    if ((size_t)t >= free_.size()) {
      free_.resize((size_t)t + 1);
      held_.resize((size_t)t + 1);
    }
  }

  int64_t block_size_;
  int64_t nblocks_;
  int64_t capacity_bytes_;
  // Backing storage over-aligned to the block size so every allocation
  // offset (a block multiple) delivers block_size alignment for user data.
  struct AlignedDeleter {
    size_t align = alignof(std::max_align_t);
    void operator()(std::byte* p) const {
      ::operator delete(p, std::align_val_t(align));
    }
  };
  std::unique_ptr<std::byte[], AlignedDeleter> backing_;
  std::vector<TaskId> owner_;  // per block: -1 unowned, else task
  std::vector<std::vector<FreeRun>> free_;  // per task
  std::vector<std::vector<FreeRun>> held_;  // per task (handover staging)
  std::vector<AllocRecord> records_;
  std::vector<uint32_t> free_slots_;
  mutable std::recursive_mutex mu_;
};

}  // namespace keel::rt
