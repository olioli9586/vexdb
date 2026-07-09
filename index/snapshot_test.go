package index

import (
	"bytes"
	"fmt"
	"math/rand"
	"testing"
)

// A snapshot round-trip must produce identical search results.
func TestSnapshotRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	h := NewHNSW(16, 200, 64)
	for i := 0; i < 500; i++ {
		mustAdd(t, h, fmt.Sprintf("v%d", i), randVec(rng, 16))
	}

	var buf bytes.Buffer
	if err := h.Save(&buf); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadHNSW(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Len() != h.Len() {
		t.Fatalf("len mismatch: %d vs %d", loaded.Len(), h.Len())
	}

	for i := 0; i < 20; i++ {
		q := randVec(rng, 16)
		a, err := h.Search(q, 5)
		if err != nil {
			t.Fatal(err)
		}
		b, err := loaded.Search(q, 5)
		if err != nil {
			t.Fatal(err)
		}
		for j := range a {
			if a[j].ID != b[j].ID {
				t.Fatalf("query %d rank %d: %s vs %s", i, j, a[j].ID, b[j].ID)
			}
		}
	}

	// The loaded index must also accept new inserts.
	if err := loaded.Add("new", randVec(rng, 16)); err != nil {
		t.Fatalf("add after load: %v", err)
	}
}
