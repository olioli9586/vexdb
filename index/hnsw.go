package index

import (
	"container/heap"
	"math"
	"math/rand"
	"sort"
	"sync"
)

// HNSW is a Hierarchical Navigable Small World graph (Malkov & Yashunin,
// 2016): approximate nearest-neighbor search in O(log n) hops instead of a
// full scan.
//
// The intuition: build a multi-layer graph where the top layers are sparse
// "highways" (few nodes, long-range links) and layer 0 contains every node
// with short-range links. A query greedily descends — coarse navigation on
// the highways, then a beam search (width ef) at the ground layer.
//
// Parameters:
//   - M:  max links per node per layer (layer 0 allows 2M). Higher = better
//     recall, more memory.
//   - efConstruction: beam width while building. Higher = better graph,
//     slower inserts.
//   - efSearch: beam width while querying. The recall/latency dial.
type HNSW struct {
	M              int
	EfConstruction int
	EfSearch       int

	mu       sync.RWMutex
	dims     int
	nodes    []*hnswNode
	byID     map[string]uint32
	entry    uint32 // entry point: a node on the top layer
	maxLevel int
	mL       float64 // level-assignment multiplier: 1/ln(M)
	rng      *rand.Rand
}

type hnswNode struct {
	id    string
	vec   []float32
	links [][]uint32 // links[l] = neighbor indices at layer l; len(links) = node's level+1
}

func NewHNSW(m, efConstruction, efSearch int) *HNSW {
	return &HNSW{
		M:              m,
		EfConstruction: efConstruction,
		EfSearch:       efSearch,
		byID:           make(map[string]uint32),
		maxLevel:       -1,
		mL:             1 / math.Log(float64(m)),
		rng:            rand.New(rand.NewSource(42)), // deterministic builds for reproducible tests
	}
}

// randomLevel draws from a geometric-like distribution: most nodes live only
// on layer 0; each higher layer holds ~1/M of the one below.
func (h *HNSW) randomLevel() int {
	return int(math.Floor(-math.Log(h.rng.Float64()) * h.mL))
}

func (h *HNSW) maxLinks(level int) int {
	if level == 0 {
		return 2 * h.M
	}
	return h.M
}

func (h *HNSW) Add(id string, vec []float32) error {
	if id == "" {
		return ErrEmptyID
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if _, ok := h.byID[id]; ok {
		return ErrDuplicateID
	}
	if h.dims == 0 {
		h.dims = len(vec)
	} else if len(vec) != h.dims {
		return ErrDimMismatch
	}
	v := make([]float32, len(vec))
	copy(v, vec)
	if err := Normalize(v); err != nil {
		return err
	}

	level := h.randomLevel()
	node := &hnswNode{id: id, vec: v, links: make([][]uint32, level+1)}
	idx := uint32(len(h.nodes))
	h.nodes = append(h.nodes, node)
	h.byID[id] = idx

	if len(h.nodes) == 1 {
		h.entry = idx
		h.maxLevel = level
		return nil
	}

	// Phase 1: greedy descent through layers above the new node's level.
	ep := h.entry
	for l := h.maxLevel; l > level; l-- {
		ep = h.greedyClosest(v, ep, l)
	}

	// Phase 2: on each shared layer, beam-search for candidates, connect to
	// a *diverse* selection of them, and prune neighbors that now exceed
	// their link budget.
	for l := min(level, h.maxLevel); l >= 0; l-- {
		cands := h.searchLayer(v, ep, h.EfConstruction, l)
		for _, c := range h.selectNeighbors(cands, h.M) {
			node.links[l] = append(node.links[l], c.idx)
			nb := h.nodes[c.idx]
			nb.links[l] = append(nb.links[l], idx)
			if len(nb.links[l]) > h.maxLinks(l) {
				h.pruneLinks(nb, l)
			}
		}
		ep = cands[0].idx
	}

	if level > h.maxLevel {
		h.maxLevel = level
		h.entry = idx
	}
	return nil
}

// selectNeighbors implements the paper's diversity heuristic (Algorithm 4):
// walk candidates closest-first and keep one only if it is closer to the
// base point than to every neighbor already kept. Plain "closest M" produces
// tight local clusters whose recall collapses at scale (measured here:
// 0.49 recall@10 on 20K×128d); diversity preserves the long-range links that
// make the graph navigable (0.99+ on the same data). If diversity rejects
// too many, backfill with the closest rejects so nodes keep m links.
func (h *HNSW) selectNeighbors(cands []searchItem, m int) []searchItem {
	if len(cands) <= m {
		return cands
	}
	selected := make([]searchItem, 0, m)
	var discarded []searchItem
	for _, c := range cands {
		if len(selected) == m {
			break
		}
		diverse := true
		for _, s := range selected {
			if distance(h.nodes[c.idx].vec, h.nodes[s.idx].vec) < c.dist {
				diverse = false
				break
			}
		}
		if diverse {
			selected = append(selected, c)
		} else {
			discarded = append(discarded, c)
		}
	}
	for len(selected) < m && len(discarded) > 0 {
		selected = append(selected, discarded[0])
		discarded = discarded[1:]
	}
	return selected
}

// pruneLinks re-selects a neighbor's links with the same diversity heuristic
// once it exceeds its budget.
func (h *HNSW) pruneLinks(n *hnswNode, level int) {
	links := n.links[level]
	cands := make([]searchItem, len(links))
	for i, l := range links {
		cands[i] = searchItem{l, distance(n.vec, h.nodes[l].vec)}
	}
	sort.Slice(cands, func(a, b int) bool { return cands[a].dist < cands[b].dist })
	sel := h.selectNeighbors(cands, h.maxLinks(level))
	out := make([]uint32, len(sel))
	for i, s := range sel {
		out[i] = s.idx
	}
	n.links[level] = out
}

func (h *HNSW) Search(vec []float32, k int) ([]Result, error) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	if len(h.nodes) == 0 {
		return nil, nil
	}
	if len(vec) != h.dims {
		return nil, ErrDimMismatch
	}
	q := make([]float32, len(vec))
	copy(q, vec)
	if err := Normalize(q); err != nil {
		return nil, err
	}

	ep := h.entry
	for l := h.maxLevel; l > 0; l-- {
		ep = h.greedyClosest(q, ep, l)
	}
	ef := max(h.EfSearch, k)
	cands := h.searchLayer(q, ep, ef, 0)

	n := min(k, len(cands))
	results := make([]Result, n)
	for i := 0; i < n; i++ {
		results[i] = Result{ID: h.nodes[cands[i].idx].id, Score: 1 - cands[i].dist}
	}
	return results, nil
}

func (h *HNSW) Len() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.nodes)
}

// greedyClosest walks one layer, always moving to the closest neighbor,
// until no neighbor improves — the "highway navigation" step.
func (h *HNSW) greedyClosest(q []float32, ep uint32, level int) uint32 {
	cur, curDist := ep, distance(q, h.nodes[ep].vec)
	for improved := true; improved; {
		improved = false
		for _, nb := range h.nodes[cur].links[level] {
			if d := distance(q, h.nodes[nb].vec); d < curDist {
				cur, curDist = nb, d
				improved = true
			}
		}
	}
	return cur
}

// searchLayer is the beam search at one layer: expand the closest unexpanded
// candidate, keep the ef best seen. Returns candidates sorted closest-first.
func (h *HNSW) searchLayer(q []float32, ep uint32, ef, level int) []searchItem {
	d := distance(q, h.nodes[ep].vec)
	visited := map[uint32]struct{}{ep: {}}
	candidates := &minHeap{{ep, d}} // frontier: closest first
	results := &maxHeap{{ep, d}}    // best ef so far: worst on top for O(1) eviction

	for candidates.Len() > 0 {
		c := heap.Pop(candidates).(searchItem)
		if c.dist > (*results)[0].dist && results.Len() >= ef {
			break // frontier is now worse than everything we keep: done
		}
		for _, nb := range h.nodes[c.idx].links[level] {
			if _, seen := visited[nb]; seen {
				continue
			}
			visited[nb] = struct{}{}
			dn := distance(q, h.nodes[nb].vec)
			if results.Len() < ef || dn < (*results)[0].dist {
				heap.Push(candidates, searchItem{nb, dn})
				heap.Push(results, searchItem{nb, dn})
				if results.Len() > ef {
					heap.Pop(results)
				}
			}
		}
	}

	out := make([]searchItem, results.Len())
	copy(out, *results)
	sort.Slice(out, func(a, b int) bool { return out[a].dist < out[b].dist })
	return out
}

type searchItem struct {
	idx  uint32
	dist float32
}

type minHeap []searchItem

func (p minHeap) Len() int            { return len(p) }
func (p minHeap) Less(i, j int) bool  { return p[i].dist < p[j].dist }
func (p minHeap) Swap(i, j int)       { p[i], p[j] = p[j], p[i] }
func (p *minHeap) Push(x any)         { *p = append(*p, x.(searchItem)) }
func (p *minHeap) Pop() any           { old := *p; n := len(old); x := old[n-1]; *p = old[:n-1]; return x }

type maxHeap []searchItem

func (p maxHeap) Len() int            { return len(p) }
func (p maxHeap) Less(i, j int) bool  { return p[i].dist > p[j].dist }
func (p maxHeap) Swap(i, j int)       { p[i], p[j] = p[j], p[i] }
func (p *maxHeap) Push(x any)         { *p = append(*p, x.(searchItem)) }
func (p *maxHeap) Pop() any           { old := *p; n := len(old); x := old[n-1]; *p = old[:n-1]; return x }
