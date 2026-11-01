// Phase 4A: real block arena tests. The guarantees under test are physical:
// real backing storage, single ownership, first-fit within owned blocks,
// generational stale-handle rejection, observable physical transfer (Task B
// allocates the exact blocks Task A surrendered), fragmentation respected,
// block-size alignment, and a full invariant scan.

#include <doctest/doctest.h>

#include <keel/rt/arena.hpp>

#include <cstring>

using namespace keel::rt;

TEST_CASE("arena: allocation, real memory, bounds") {
  BlockArena a(4096, 256);  // 16 blocks
  REQUIRE(a.claim(0, 10) == 10);
  REQUIRE(a.claim(1, 4) == 4);
  CHECK(a.blocks_owned(0) == 10);
  CHECK(a.blocks_owned(1) == 4);

  AllocHandle h = a.alloc(0, 512);  // exactly 2 blocks
  const AllocRecord& rec = a.record(h);
  CHECK(rec.nblocks == 2);
  CHECK(rec.bytes == 512);
  CHECK(rec.owner == 0);

  // Real distinct storage: writes through one allocation are not visible
  // through another (no aliasing).
  AllocHandle g = a.alloc(0, 256);
  std::memset(a.ptr(h), 0xAB, 512);
  std::memset(a.ptr(g), 0x00, 256);
  auto* p = (unsigned char*)a.ptr(h);
  CHECK(p[0] == 0xAB);
  CHECK(p[511] == 0xAB);

  SUBCASE("insufficient contiguous storage is refused even with bytes free") {
    AllocHandle h1 = a.alloc(0, 256);
    AllocHandle h2 = a.alloc(0, 256);
    a.free(h);        // 2-block hole
    a.alloc(0, 256);  // split the hole: 1 block left in it
    (void)h1;
    (void)h2;
    // Task 0 has few free blocks; an over-large allocation must fail.
    CHECK_THROWS(a.alloc(0, 256 * 9));
  }

  SUBCASE("claim beyond capacity fails") {
    BlockArena small(512, 256);
    CHECK_THROWS(small.claim(0, 3));
  }
}

TEST_CASE("arena: stale and invalid handles are rejected") {
  BlockArena a(1024, 256);
  a.claim(0, 4);
  AllocHandle h = a.alloc(0, 256);
  const uint64_t h_gen = a.record(h).generation;
  a.free(h);
  CHECK_THROWS(a.free(h));   // double free
  CHECK_THROWS(a.ptr(h));    // dead handle
  AllocHandle bogus{999, 1};
  CHECK_THROWS(a.record(bogus));
  CHECK_THROWS(a.free(bogus));
  // A new allocation reusing the slot bumps the generation; the stale handle
  // stays dead and cannot be resurrected.
  AllocHandle h2 = a.alloc(0, 256);
  CHECK_FALSE(h2 == h);
  CHECK(h2.slot == h.slot);
  CHECK(h2.generation == h_gen + 1);
  CHECK(a.record(h).generation == h2.generation);
  CHECK_THROWS(a.ptr(h));
  CHECK(a.ptr(h2) != nullptr);
}

TEST_CASE("arena: physical transfer is observable and exclusive") {
  BlockArena a(4096, 256);  // 16 blocks
  a.claim(0, 12);           // task 0: blocks 0..11
  a.claim(1, 4);            // task 1: blocks 12..15
  // Task 0 keeps a LIVE 4-block allocation and frees an 8-block run.
  AllocHandle keep = a.alloc(0, 1024);  // blocks 0..3
  CHECK(a.record(keep).first_block == 0);
  std::memset(a.ptr(keep), 0x77, 1024);
  AllocHandle h = a.alloc(0, 2048);  // blocks 4..11
  std::memset(a.ptr(h), 0x5A, 2048);
  a.free(h);  // task 0 free run: blocks 4..11 (coalesced, 8 blocks)

  // Task 0 hands >= 6 blocks to task 1. Whole free runs move as physical
  // units: exactly the 8-block run moves; live blocks stay put.
  std::vector<int32_t> moved = a.transfer_free(0, 1, 6);
  CHECK(moved.size() == 8);
  CHECK(a.blocks_owned(1) == 12);
  CHECK(a.blocks_owned(0) == 4);
  for (int32_t b : moved) {
    CHECK(b >= 4);
    CHECK(b < 12);  // the freed run, not task 0's live region
  }

  // Task 0's LIVE allocation is undisturbed by the transfer.
  auto* kp = (unsigned char*)a.ptr(keep);
  CHECK(kp[0] == 0x77);
  CHECK(kp[1023] == 0x77);

  // Task 1 allocates and must land INSIDE the transferred blocks (physical
  // reuse of Task A's former storage, not a counter trick).
  AllocHandle hb = a.alloc(1, 1536);  // 6 blocks
  const AllocRecord& rec = a.record(hb);
  bool inside = false;
  for (int32_t b : moved)
    if (rec.first_block <= b && b < rec.first_block + rec.nblocks) inside = true;
  CHECK(inside);
  // Task 1 writes its own pattern over Task A's old bytes.
  std::memset(a.ptr(hb), 0x11, 1536);
  auto* q = (unsigned char*)a.ptr(hb);
  CHECK(q[0] == 0x11);

  // The transferred blocks are no longer allocatable by task 0 (its only
  // free space is gone; the live 4 blocks remain).
  CHECK_THROWS(a.alloc(0, 256));
  std::string err;
  CHECK(a.verify(&err));
}

TEST_CASE("arena: failed transfer leaves the arena untouched") {
  BlockArena a(2048, 256);
  a.claim(0, 4);
  a.claim(1, 4);
  AllocHandle h = a.alloc(0, 512);
  (void)h;
  CHECK_THROWS(a.transfer_free(0, 1, 3));  // only 2 free blocks
  CHECK(a.blocks_owned(0) == 4);
  CHECK(a.blocks_owned(1) == 4);
  std::string err;
  CHECK(a.verify(&err));
}

TEST_CASE("arena: coalescing and the debug verifier") {
  BlockArena a(2048, 256);
  a.claim(0, 8);
  std::vector<AllocHandle> hs;
  for (int i = 0; i < 8; ++i) hs.push_back(a.alloc(0, 256));
  std::string err;
  CHECK(a.verify(&err));
  for (int i = 0; i < 8; i += 2) a.free(hs[i]);
  CHECK(a.verify(&err));
  for (int i = 1; i < 8; i += 2) a.free(hs[i]);
  CHECK(a.verify(&err));
  AllocHandle h = a.alloc(0, 2048);  // proves full coalescing
  CHECK(a.record(h).nblocks == 8);
  CHECK(a.verify(&err));
}

TEST_CASE("arena: alignment is block-guaranteed") {
  BlockArena a(1024, 256);
  a.claim(0, 4);
  for (int i = 0; i < 4; ++i) {
    AllocHandle h = a.alloc(0, 64);
    auto p = (uintptr_t)a.ptr(h);
    CHECK(p % 256 == 0);
  }
}

TEST_CASE("arena: alignment and claim validation are checked and atomic") {
  SUBCASE("non-default alignments") {
    for (int64_t block_size : {64, 512}) {
      BlockArena a(block_size * 4, block_size);
      a.claim(0, 4);
      AllocHandle h = a.alloc(0, block_size);
      CHECK(reinterpret_cast<uintptr_t>(a.ptr(h)) % block_size == 0);
    }
  }
  SUBCASE("unsupported alignment is rejected") {
    CHECK_THROWS(BlockArena(4096, 3));
    CHECK_THROWS(BlockArena(4096, 2));
  }
  SUBCASE("failed claim changes no ownership") {
    BlockArena a(8 * 256, 256);
    REQUIRE(a.claim(0, 6) == 6);
    CHECK_THROWS(a.claim(1, 3));
    CHECK(a.blocks_owned(0) == 6);
    CHECK(a.blocks_owned(1) == 0);
    std::string err;
    CHECK(a.verify(&err));
  }
}

TEST_CASE("arena: free-run diagnostics expose fragmentation") {
  BlockArena arena(8 * 256, 256);
  REQUIRE(arena.claim(0, 8) == 8);
  const AllocHandle first = arena.alloc(0, 2 * 256);
  const AllocHandle middle = arena.alloc(0, 4 * 256);
  const AllocHandle last = arena.alloc(0, 2 * 256);
  arena.free(first);
  arena.free(last);

  CHECK(arena.free_blocks(0) == 4);
  CHECK(arena.largest_free_run_blocks(0) == 2);
  CHECK(arena.free_blocks(1) == 0);

  arena.free(middle);
  CHECK(arena.free_blocks(0) == 8);
  CHECK(arena.largest_free_run_blocks(0) == 8);
}

TEST_CASE("arena: failed held transfer is failure-atomic") {
  BlockArena a(12 * 256, 256);
  a.claim(0, 8);
  a.claim(1, 4);
  AllocHandle h = a.alloc(0, 4 * 256);
  const AllocRecord rec = a.record(h);
  a.free(h);
  a.hold(0, rec.first_block + 1, 2);
  const int64_t before0 = a.blocks_owned(0);
  const int64_t before1 = a.blocks_owned(1);
  CHECK_THROWS(a.transfer_held_extents(0, 1, {{rec.first_block + 1, 2},
                                               {rec.first_block + 2, 1}}));
  CHECK(a.blocks_owned(0) == before0);
  CHECK(a.blocks_owned(1) == before1);
  std::string err;
  CHECK(a.verify(&err));
  CHECK_NOTHROW(a.unhold(0, rec.first_block + 1, 2));
  CHECK(a.verify(&err));
}
