package index

import (
	"encoding/gob"
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

// LoadHNSW reconstructs an index from a snapshot written by Save.
func LoadHNSW(r io.Reader) (*HNSW, error) {
	var snap hnswSnapshot
	if err := gob.NewDecoder(r).Decode(&snap); err != nil {
		return nil, err
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
