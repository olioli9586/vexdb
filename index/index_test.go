package index

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"testing"
)

func TestFlatExactOrdering(t *testing.T) {
	f := NewFlat()
	mustAdd(t, f, "east", []float32{1, 0})
	mustAdd(t, f, "north", []float32{0, 1})
	mustAdd(t, f, "northeast", []float32{1, 1})

	got, err := f.Search([]float32{2, 0.1}, 3)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"east", "northeast", "north"}
	for i, w := range want {
		if got[i].ID != w {
			t.Fatalf("rank %d: got %s want %s (results %v)", i, got[i].ID, w, got)
		}
	}
	if got[0].Score < 0.99 {
		t.Fatalf("near-identical vector should score ~1, got %f", got[0].Score)
	}
}

func TestValidation(t *testing.T) {
	for _, idx := range []Index{NewFlat(), NewHNSW(16, 200, 64)} {
		if err := idx.Add("", []float32{1}); err != ErrEmptyID {
			t.Fatalf("want ErrEmptyID, got %v", err)
		}
		if err := idx.Add("a", []float32{0, 0}); err != ErrZeroVector {
			t.Fatalf("want ErrZeroVector, got %v", err)
		}
		mustAdd(t, idx, "a", []float32{1, 2})
		if err := idx.Add("a", []float32{1, 2}); err != ErrDuplicateID {
			t.Fatalf("want ErrDuplicateID, got %v", err)
		}
		if err := idx.Add("b", []float32{1, 2, 3}); err != ErrDimMismatch {
			t.Fatalf("want ErrDimMismatch, got %v", err)
		}
		if _, err := idx.Search([]float32{1}, 5); err != ErrDimMismatch {
			t.Fatalf("want ErrDimMismatch on search, got %v", err)
		}
	}
}

// TestHNSWRecall builds both indexes over the same random vectors and
// checks HNSW's recall@10 against Flat's exact ground truth.
func TestHNSWRecall(t *testing.T) {
	const (
		n       = 2000
		dims    = 32
		queries = 50
		k       = 10
	)
	rng := rand.New(rand.NewSource(1))
	flat := NewFlat()
	hnsw := NewHNSW(16, 200, 64)
	for i := 0; i < n; i++ {
		v := randVec(rng, dims)
		id := fmt.Sprintf("v%d", i)
		mustAdd(t, flat, id, v)
		mustAdd(t, hnsw, id, v)
	}

	var hits, total int
	for i := 0; i < queries; i++ {
		q := randVec(rng, dims)
		truth, err := flat.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		approx, err := hnsw.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		truthSet := map[string]struct{}{}
		for _, r := range truth {
			truthSet[r.ID] = struct{}{}
		}
		for _, r := range approx {
			if _, ok := truthSet[r.ID]; ok {
				hits++
			}
		}
		total += k
	}
	recall := float64(hits) / float64(total)
	t.Logf("recall@%d = %.3f over %d queries", k, recall, queries)
	if recall < 0.9 {
		t.Fatalf("recall@%d = %.3f, want >= 0.9", k, recall)
	}
}

func mustAdd(t *testing.T, idx Index, id string, v []float32) {
	t.Helper()
	if err := idx.Add(id, v); err != nil {
		t.Fatalf("add %s: %v", id, err)
	}
}

func randVec(rng *rand.Rand, dims int) []float32 {
	v := make([]float32, dims)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	return v
}

// A rejected first insert must not pin the index's dimensionality: nothing
// was stored, so the next valid vector of any length should be accepted.
func TestRejectedFirstVectorDoesNotFixDims(t *testing.T) {
	for _, idx := range []Index{NewFlat(), NewHNSW(16, 200, 64)} {
		if err := idx.Add("zero", []float32{0, 0, 0}); err != ErrZeroVector {
			t.Fatalf("%T: want ErrZeroVector, got %v", idx, err)
		}
		if err := idx.Add("nan", []float32{float32(math.NaN()), 1, 2, 3}); err != ErrNonFinite {
			t.Fatalf("%T: want ErrNonFinite, got %v", idx, err)
		}
		mustAdd(t, idx, "a", []float32{1, 2})
		got, err := idx.Search([]float32{1, 2}, 1)
		if err != nil {
			t.Fatalf("%T: %v", idx, err)
		}
		if len(got) != 1 || got[0].ID != "a" {
			t.Fatalf("%T: got %v, want [a]", idx, got)
		}
	}
}

// Search with k <= 0 used to panic (negative slice bound in Flat, negative
// makeslice in HNSW). It should return no results instead.
func TestSearchNonPositiveK(t *testing.T) {
	for _, idx := range []Index{NewFlat(), NewHNSW(16, 200, 64)} {
		mustAdd(t, idx, "a", []float32{1, 0})
		mustAdd(t, idx, "b", []float32{0, 1})
		for _, k := range []int{0, -1, -100} {
			got, err := idx.Search([]float32{1, 0}, k)
			if err != nil {
				t.Fatalf("%T k=%d: %v", idx, k, err)
			}
			if len(got) != 0 {
				t.Fatalf("%T k=%d: got %v, want none", idx, k, got)
			}
		}
	}
}

// Subnormal float32 inputs have a norm whose reciprocal overflows float32.
// Normalizing in float32 turned [1e-45, 0] into [+Inf, NaN], which then
// poisoned every score it touched (and broke JSON encoding of results).
func TestNormalizeSubnormal(t *testing.T) {
	v := []float32{1e-45, 0}
	if err := Normalize(v); err != nil {
		t.Fatal(err)
	}
	if v[0] != 1 || v[1] != 0 {
		t.Fatalf("got %v, want [1 0]", v)
	}

	for _, idx := range []Index{NewFlat(), NewHNSW(16, 200, 64)} {
		mustAdd(t, idx, "tiny", []float32{1e-45, 0})
		mustAdd(t, idx, "north", []float32{0, 1})
		got, err := idx.Search([]float32{1, 0}, 2)
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range got {
			if math.IsNaN(float64(r.Score)) || math.IsInf(float64(r.Score), 0) {
				t.Fatalf("%T: non-finite score in %v", idx, got)
			}
		}
		if got[0].ID != "tiny" || got[0].Score < 0.99 {
			t.Fatalf("%T: got %v, want tiny first with score ~1", idx, got)
		}
	}
}

func TestNonFiniteRejected(t *testing.T) {
	inf := float32(math.Inf(1))
	nan := float32(math.NaN())
	for _, idx := range []Index{NewFlat(), NewHNSW(16, 200, 64)} {
		for _, v := range [][]float32{{inf, 1}, {1, -inf}, {nan, 1}} {
			if err := idx.Add("x", v); err != ErrNonFinite {
				t.Fatalf("%T add %v: want ErrNonFinite, got %v", idx, v, err)
			}
		}
		mustAdd(t, idx, "a", []float32{1, 0})
		if _, err := idx.Search([]float32{nan, 1}, 1); err != ErrNonFinite {
			t.Fatalf("%T search: want ErrNonFinite, got %v", idx, err)
		}
		if idx.Len() != 1 {
			t.Fatalf("%T: rejected vectors were stored (len %d)", idx, idx.Len())
		}
	}
}

// M < 2 makes the level multiplier 1/ln(M) infinite (M=1) or degenerate
// (M=0): M=1 panicked on the first inserts and M=0 built a graph with no
// links. Such values are clamped to the minimum usable M.
func TestHNSWDegenerateM(t *testing.T) {
	for _, m := range []int{-1, 0, 1} {
		h := NewHNSW(m, 50, 16)
		if h.M < 2 {
			t.Fatalf("M=%d: not clamped (got %d)", m, h.M)
		}
		rng := rand.New(rand.NewSource(5))
		vecs := make([][]float32, 200)
		for i := range vecs {
			vecs[i] = randVec(rng, 8)
			mustAdd(t, h, fmt.Sprintf("v%d", i), vecs[i])
		}
		// M=2 is a sparse graph, so demand a working one rather than a
		// perfect one: full result lists and most self-lookups found.
		hits := 0
		for i, v := range vecs {
			got, err := h.Search(v, 5)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != 5 {
				t.Fatalf("M=%d: query v%d got %d results, want 5", m, i, len(got))
			}
			if got[0].ID == fmt.Sprintf("v%d", i) {
				hits++
			}
		}
		if hits < 180 {
			t.Fatalf("M=%d: only %d/200 self-lookups found", m, hits)
		}
	}
}

// zeroSource makes rand.Float64 return exactly 0, the one draw for which
// -ln(u) is +Inf.
type zeroSource struct{}

func (zeroSource) Int63() int64 { return 0 }
func (zeroSource) Seed(int64)   {}

func TestRandomLevelZeroDraw(t *testing.T) {
	h := NewHNSW(16, 200, 64)
	h.rng = rand.New(zeroSource{})
	lvl := h.randomLevel()
	if lvl < 0 || lvl > 1000 {
		t.Fatalf("randomLevel with u=0 = %d, want a small non-negative level", lvl)
	}
	// And the index must still work when such a node is inserted.
	mustAdd(t, h, "a", []float32{1, 0})
	mustAdd(t, h, "b", []float32{0, 1})
	got, err := h.Search([]float32{0, 1}, 1)
	if err != nil || len(got) != 1 || got[0].ID != "b" {
		t.Fatalf("got %v, %v", got, err)
	}
}

// Concurrent readers and writers must not race (run with -race).
func TestHNSWConcurrentAddSearch(t *testing.T) {
	h := NewHNSW(8, 50, 16)
	rng := rand.New(rand.NewSource(9))
	mustAdd(t, h, "seed", randVec(rng, 8))
	queries := make([][]float32, 50)
	for i := range queries {
		queries[i] = randVec(rng, 8)
	}
	inserts := make([][]float32, 200)
	for i := range inserts {
		inserts[i] = randVec(rng, 8)
	}

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		for i, v := range inserts {
			if err := h.Add(fmt.Sprintf("v%d", i), v); err != nil {
				t.Error(err)
				return
			}
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < 400; i++ {
			if _, err := h.Search(queries[i%len(queries)], 5); err != nil {
				t.Error(err)
				return
			}
		}
	}()
	wg.Wait()
	if h.Len() != 201 {
		t.Fatalf("len = %d, want 201", h.Len())
	}
}
