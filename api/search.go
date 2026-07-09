// Package handler is the Vercel serverless function behind the playground:
// typo-tolerant word search over a pre-built HNSW snapshot.
//
// The snapshot (15K dictionary words as character-trigram vectors) is
// embedded in the binary and loaded once per instance — this is exactly what
// the snapshot format exists for: cold starts deserialize the graph in
// ~100ms instead of rebuilding it for seconds.
package handler

import (
	_ "embed"
	"bytes"
	"encoding/json"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/olioli9586/vexdb/feature"
	"github.com/olioli9586/vexdb/index"
)

//go:embed demo.snap
var snapshot []byte

var (
	once sync.Once
	idx  *index.HNSW
	err  error
)

func Handler(w http.ResponseWriter, r *http.Request) {
	once.Do(func() {
		idx, err = index.LoadHNSW(bytes.NewReader(snapshot))
	})
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]string{"error": "snapshot failed to load"})
		return
	}

	q := strings.TrimSpace(r.URL.Query().Get("q"))
	if q == "" || len(q) > 40 {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "pass ?q=word (max 40 chars)"})
		return
	}

	start := time.Now()
	results, serr := idx.Search(feature.Trigram(q), 10)
	if serr != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": serr.Error()})
		return
	}
	json.NewEncoder(w).Encode(map[string]any{
		"query":      q,
		"results":    results,
		"latency_us": time.Since(start).Microseconds(),
		"indexed":    idx.Len(),
	})
}
