// bench compares Flat (exact) vs HNSW (approximate): build time, query
// latency, and recall across a sweep of efSearch values.
//
// Two data modes:
//   - random:    i.i.d. gaussian vectors — the worst case for ANN search
//     (in high dimensions all similarities concentrate, so there is no
//     structure to exploit).
//   - clustered: vectors drawn around cluster centers — the shape of real
//     embedding data, whose intrinsic dimension is far below its raw
//     dimension. Recall at a given ef is dramatically higher here.
package main

import (
	"flag"
	"fmt"
	"math/rand"
	"time"

	"github.com/olioli9586/vexdb/index"
)

func main() {
	n := flag.Int("n", 20000, "number of vectors")
	dims := flag.Int("dims", 128, "vector dimensions")
	queries := flag.Int("queries", 100, "number of test queries")
	k := flag.Int("k", 10, "neighbors per query")
	clustered := flag.Bool("clustered", false, "clustered data (realistic) instead of uniform random (adversarial)")
	flag.Parse()

	rng := rand.New(rand.NewSource(7))
	gen := newGenerator(rng, *dims, *clustered)
	vecs := make([][]float32, *n)
	for i := range vecs {
		vecs[i] = gen()
	}
	qs := make([][]float32, *queries)
	for i := range qs {
		qs[i] = gen()
	}

	mode := "random (adversarial)"
	if *clustered {
		mode = "clustered (realistic)"
	}
	fmt.Printf("dataset: %d vectors × %d dims, %s, %d queries, k=%d\n\n", *n, *dims, mode, *queries, *k)

	flat := index.NewFlat()
	buildFlat := buildInto(flat, vecs)
	hnsw := index.NewHNSW(16, 200, 64)
	buildHNSW := buildInto(hnsw, vecs)
	fmt.Printf("build: flat %s · hnsw %s\n\n", buildFlat.Round(time.Millisecond), buildHNSW.Round(time.Millisecond))

	truth := make([][]index.Result, *queries)
	flatTime := timeQueries(flat, qs, *k, truth)
	fmt.Printf("%-12s %14s %10s %10s\n", "index", "avg query", "qps", "recall@k")
	fmt.Printf("%-12s %14s %10.0f %10s\n", "flat", (flatTime / time.Duration(*queries)).Round(time.Microsecond),
		float64(*queries)/flatTime.Seconds(), "1.000")

	for _, ef := range []int{32, 64, 128, 256, 512} {
		hnsw.EfSearch = ef
		approx := make([][]index.Result, *queries)
		t := timeQueries(hnsw, qs, *k, approx)
		fmt.Printf("%-12s %14s %10.0f %10.3f\n", fmt.Sprintf("hnsw ef=%d", ef),
			(t / time.Duration(*queries)).Round(time.Microsecond),
			float64(*queries)/t.Seconds(), recall(truth, approx, *k))
	}
}

func recall(truth, approx [][]index.Result, k int) float64 {
	var hits int
	for i := range truth {
		set := map[string]struct{}{}
		for _, r := range truth[i] {
			set[r.ID] = struct{}{}
		}
		for _, r := range approx[i] {
			if _, ok := set[r.ID]; ok {
				hits++
			}
		}
	}
	return float64(hits) / float64(len(truth)*k)
}

// newGenerator returns a vector source. Clustered mode places 200 gaussian
// centers and samples tight around them — a crude stand-in for the manifold
// structure of real embeddings.
func newGenerator(rng *rand.Rand, dims int, clustered bool) func() []float32 {
	if !clustered {
		return func() []float32 { return randVec(rng, dims, 1) }
	}
	const numClusters = 200
	centers := make([][]float32, numClusters)
	for i := range centers {
		centers[i] = randVec(rng, dims, 1)
	}
	return func() []float32 {
		c := centers[rng.Intn(numClusters)]
		v := randVec(rng, dims, 0.15) // small noise around the center
		for i := range v {
			v[i] += c[i]
		}
		return v
	}
}

func buildInto(idx index.Index, vecs [][]float32) time.Duration {
	start := time.Now()
	for i, v := range vecs {
		if err := idx.Add(fmt.Sprintf("v%d", i), v); err != nil {
			panic(err)
		}
	}
	return time.Since(start)
}

func timeQueries(idx index.Index, qs [][]float32, k int, out [][]index.Result) time.Duration {
	start := time.Now()
	for i, q := range qs {
		r, err := idx.Search(q, k)
		if err != nil {
			panic(err)
		}
		out[i] = r
	}
	return time.Since(start)
}

func randVec(rng *rand.Rand, dims int, scale float64) []float32 {
	v := make([]float32, dims)
	for i := range v {
		v[i] = float32(rng.NormFloat64() * scale)
	}
	return v
}
