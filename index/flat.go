package index

import (
	"sort"
	"sync"
)

// Flat is an exact brute-force index: search scans every vector. O(n·d) per
// query, but never wrong — it is both the baseline for small collections and
// the ground truth that HNSW's recall is measured against.
type Flat struct {
	mu   sync.RWMutex
	dims int
	ids  []string
	vecs [][]float32
	byID map[string]struct{}
}

func NewFlat() *Flat {
	return &Flat{byID: make(map[string]struct{})}
}

func (f *Flat) Add(id string, vec []float32) error {
	if id == "" {
		return ErrEmptyID
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if _, ok := f.byID[id]; ok {
		return ErrDuplicateID
	}
	if f.dims == 0 {
		f.dims = len(vec)
	} else if len(vec) != f.dims {
		return ErrDimMismatch
	}
	v := make([]float32, len(vec))
	copy(v, vec)
	if err := Normalize(v); err != nil {
		return err
	}
	f.ids = append(f.ids, id)
	f.vecs = append(f.vecs, v)
	f.byID[id] = struct{}{}
	return nil
}

func (f *Flat) Search(vec []float32, k int) ([]Result, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()
	if len(f.vecs) == 0 {
		return nil, nil
	}
	if len(vec) != f.dims {
		return nil, ErrDimMismatch
	}
	q := make([]float32, len(vec))
	copy(q, vec)
	if err := Normalize(q); err != nil {
		return nil, err
	}

	results := make([]Result, len(f.vecs))
	for i, v := range f.vecs {
		results[i] = Result{ID: f.ids[i], Score: dot(q, v)}
	}
	sort.Slice(results, func(a, b int) bool { return results[a].Score > results[b].Score })
	if k < len(results) {
		results = results[:k]
	}
	return results, nil
}

func (f *Flat) Len() int {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return len(f.ids)
}
