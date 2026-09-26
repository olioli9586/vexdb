package index

import (
	"encoding/gob"
	"fmt"
	"io"
	"math"
	"math/rand"
)

// Snapshot support: serialize a built HNSW graph so it can be loaded
// instantly instead of rebuilt from a WAL replay (or re-inserted vector by
// vector). This is what makes serverless/cold-start deployments practical:
// building 15K vectors takes seconds; loading their snapshot takes ~100ms.

type hnswSnapshot struct {
	M              int
	EfConstruction int
	EfSearch       int
	Dims           int
	Entry          uint32
	MaxLevel       int
	IDs            []string
	Vecs           [][]float32
	Links          [][][]uint32
}

// Save writes the full graph to w with encoding/gob.
func (h *HNSW) Save(w io.Writer) error {
	h.mu.RLock()
	defer h.mu.RUnlock()
	snap := hnswSnapshot{
		M:              h.M,
		EfConstruction: h.EfConstruction,
		EfSearch:       h.EfSearch,
		Dims:           h.dims,
		Entry:          h.entry,
		MaxLevel:       h.maxLevel,
		IDs:            make([]string, len(h.nodes)),
		Vecs:           make([][]float32, len(h.nodes)),
		Links:          make([][][]uint32, len(h.nodes)),
	}
	for i, n := range h.nodes {
		snap.IDs[i] = n.id
		snap.Vecs[i] = n.vec
		snap.Links[i] = n.links
	}
	return gob.NewEncoder(w).Encode(&snap)
}

// LoadHNSW reconstructs an index from a snapshot written by Save. The graph
// is checked for structural consistency first, so a corrupt snapshot is an
// error here rather than an index-out-of-range panic in a later search.
func LoadHNSW(r io.Reader) (*HNSW, error) {
	var snap hnswSnapshot
	if err := gob.NewDecoder(r).Decode(&snap); err != nil {
		return nil, err
	}
	if err := snap.validate(); err != nil {
		return nil, fmt.Errorf("invalid snapshot: %w", err)
	}
	if len(snap.IDs) == 0 {
		snap.Entry, snap.MaxLevel = 0, -1
	}
	h := &HNSW{
		M:              snap.M,
		EfConstruction: snap.EfConstruction,
		EfSearch:       snap.EfSearch,
		dims:           snap.Dims,
		entry:          snap.Entry,
		maxLevel:       snap.MaxLevel,
		byID:           make(map[string]uint32, len(snap.IDs)),
		mL:             1 / math.Log(float64(snap.M)),
		rng:            rand.New(rand.NewSource(42)),
		nodes:          make([]*hnswNode, len(snap.IDs)),
	}
	for i := range snap.IDs {
		h.nodes[i] = &hnswNode{id: snap.IDs[i], vec: snap.Vecs[i], links: snap.Links[i]}
		h.byID[snap.IDs[i]] = uint32(i)
	}
	return h, nil
}

// validate checks the invariants Search and Add rely on: parallel slices of
// equal length, an entry point on the top layer, vectors of the recorded
// dimensionality, unique ids, and every link pointing at a node that exists
// on that layer.
func (s *hnswSnapshot) validate() error {
	n := len(s.IDs)
	if len(s.Vecs) != n || len(s.Links) != n {
		return fmt.Errorf("%d ids but %d vectors and %d link lists", n, len(s.Vecs), len(s.Links))
	}
	if s.M < minM {
		return fmt.Errorf("M = %d, want >= %d", s.M, minM)
	}
	if n == 0 {
		return nil
	}
	if s.Dims <= 0 {
		return fmt.Errorf("dims = %d", s.Dims)
	}
	if int(s.Entry) >= n {
		return fmt.Errorf("entry point %d out of range (%d nodes)", s.Entry, n)
	}
	if s.MaxLevel != len(s.Links[s.Entry])-1 {
		return fmt.Errorf("max level %d, but the entry point has %d layers", s.MaxLevel, len(s.Links[s.Entry]))
	}
	seen := make(map[string]struct{}, n)
	for i, id := range s.IDs {
		if id == "" {
			return fmt.Errorf("node %d: empty id", i)
		}
		if _, dup := seen[id]; dup {
			return fmt.Errorf("node %d: duplicate id %q", i, id)
		}
		seen[id] = struct{}{}
		if len(s.Vecs[i]) != s.Dims {
			return fmt.Errorf("node %d: %d dims, want %d", i, len(s.Vecs[i]), s.Dims)
		}
		if len(s.Links[i]) == 0 || len(s.Links[i])-1 > s.MaxLevel {
			return fmt.Errorf("node %d: %d layers, want 1..%d", i, len(s.Links[i]), s.MaxLevel+1)
		}
		for l, links := range s.Links[i] {
			for _, nb := range links {
				if int(nb) >= n || len(s.Links[nb]) <= l {
					return fmt.Errorf("node %d layer %d: link to %d, which is not on that layer", i, l, nb)
				}
			}
		}
	}
	return nil
}
