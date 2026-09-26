package index

import (
	"bytes"
	"encoding/gob"
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

// An empty index must round-trip and stay usable.
func TestSnapshotEmptyIndex(t *testing.T) {
	var buf bytes.Buffer
	if err := NewHNSW(16, 200, 64).Save(&buf); err != nil {
		t.Fatal(err)
	}
	h, err := LoadHNSW(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if h.Len() != 0 {
		t.Fatalf("len = %d, want 0", h.Len())
	}
	if got, err := h.Search([]float32{1, 0}, 3); err != nil || len(got) != 0 {
		t.Fatalf("search empty: %v, %v", got, err)
	}
	mustAdd(t, h, "a", []float32{1, 0})
	mustAdd(t, h, "b", []float32{0, 1})
	got, err := h.Search([]float32{0, 1}, 1)
	if err != nil || len(got) != 1 || got[0].ID != "b" {
		t.Fatalf("search after add: %v, %v", got, err)
	}
}

// LoadHNSW decodes bytes from outside the process, so a structurally
// inconsistent snapshot must be rejected with an error instead of
// panicking on load or on the first search.
func TestLoadHNSWRejectsMalformed(t *testing.T) {
	fresh := func(t *testing.T) hnswSnapshot {
		h := NewHNSW(4, 50, 16)
		rng := rand.New(rand.NewSource(11))
		for i := 0; i < 60; i++ {
			mustAdd(t, h, fmt.Sprintf("v%d", i), randVec(rng, 4))
		}
		var buf bytes.Buffer
		if err := h.Save(&buf); err != nil {
			t.Fatal(err)
		}
		var s hnswSnapshot
		if err := gob.NewDecoder(&buf).Decode(&s); err != nil {
			t.Fatal(err)
		}
		return s
	}
	// Indexes of an upper-layer node and a layer-0-only node, for the
	// "neighbor lacks the layer" case.
	layers := func(s *hnswSnapshot) (upper, ground int) {
		upper, ground = -1, -1
		for i, l := range s.Links {
			if len(l) > 1 && upper < 0 {
				upper = i
			}
			if len(l) == 1 && ground < 0 {
				ground = i
			}
		}
		return upper, ground
	}

	cases := []struct {
		name   string
		mutate func(s *hnswSnapshot)
	}{
		{"fewer vectors than ids", func(s *hnswSnapshot) { s.Vecs = s.Vecs[:len(s.Vecs)-1] }},
		{"fewer link lists than ids", func(s *hnswSnapshot) { s.Links = s.Links[:len(s.Links)-1] }},
		{"entry out of range", func(s *hnswSnapshot) { s.Entry = uint32(len(s.IDs)) }},
		{"max level above entry's level", func(s *hnswSnapshot) { s.MaxLevel = len(s.Links[s.Entry]) }},
		{"link out of range", func(s *hnswSnapshot) { s.Links[0][0] = append(s.Links[0][0], uint32(len(s.IDs))) }},
		{"neighbor lacks the layer", func(s *hnswSnapshot) {
			up, gr := layers(s)
			s.Links[up][1] = append(s.Links[up][1], uint32(gr))
		}},
		{"node with no layers", func(s *hnswSnapshot) { s.Links[3] = nil }},
		{"vector with wrong dims", func(s *hnswSnapshot) { s.Vecs[2] = s.Vecs[2][:1] }},
		{"duplicate id", func(s *hnswSnapshot) { s.IDs[1] = s.IDs[0] }},
		{"empty id", func(s *hnswSnapshot) { s.IDs[5] = "" }},
		{"M below 2", func(s *hnswSnapshot) { s.M = 1 }},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			s := fresh(t)
			c.mutate(&s)
			var buf bytes.Buffer
			if err := gob.NewEncoder(&buf).Encode(&s); err != nil {
				t.Fatal(err)
			}
			var h *HNSW
			var err error
			func() {
				defer func() {
					if p := recover(); p != nil {
						t.Fatalf("LoadHNSW panicked: %v", p)
					}
				}()
				h, err = LoadHNSW(&buf)
			}()
			if err == nil {
				t.Fatalf("LoadHNSW accepted a malformed snapshot (len %d)", h.Len())
			}
		})
	}

	// Control: the unmodified snapshot loads.
	s := fresh(t)
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(&s); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadHNSW(&buf); err != nil {
		t.Fatalf("valid snapshot rejected: %v", err)
	}
}

func TestLoadHNSWTruncated(t *testing.T) {
	h := NewHNSW(16, 200, 64)
	rng := rand.New(rand.NewSource(12))
	for i := 0; i < 50; i++ {
		mustAdd(t, h, fmt.Sprintf("v%d", i), randVec(rng, 8))
	}
	var buf bytes.Buffer
	if err := h.Save(&buf); err != nil {
		t.Fatal(err)
	}
	b := buf.Bytes()
	if _, err := LoadHNSW(bytes.NewReader(b[:len(b)/2])); err == nil {
		t.Fatal("truncated snapshot loaded without error")
	}
}
