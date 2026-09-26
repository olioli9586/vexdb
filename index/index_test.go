package index

import (
	"fmt"
	"math"
	"math/rand"
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
