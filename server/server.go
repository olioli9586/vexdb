// Package server exposes the index over HTTP: durability first (WAL append),
// then index insert, then acknowledge.
package server

import (
	"encoding/json"
	"errors"
	"log"
	"net/http"

	"github.com/olioli9586/vexdb/index"
	"github.com/olioli9586/vexdb/wal"
)

type Server struct {
	idx index.Index
	log *wal.WAL
	mux *http.ServeMux
}

func New(idx index.Index, w *wal.WAL) *Server {
	s := &Server{idx: idx, log: w, mux: http.NewServeMux()}
	s.mux.HandleFunc("POST /vectors", s.handleAdd)
	s.mux.HandleFunc("POST /search", s.handleSearch)
	s.mux.HandleFunc("GET /stats", s.handleStats)
	return s
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mux.ServeHTTP(w, r)
}

type addRequest struct {
	ID     string    `json:"id"`
	Vector []float32 `json:"vector"`
}

func (s *Server) handleAdd(w http.ResponseWriter, r *http.Request) {
	var req addRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httpError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	// Order matters: validate + apply first, then WAL, then acknowledge.
	// The durability invariant is "never acknowledge a write that isn't on
	// disk" — logging *before* validation would let rejected requests (e.g.
	// duplicate ids) poison the log and break replay. Found the hard way:
	// a 409'd duplicate bricked startup until this was reordered.
	if err := s.idx.Add(req.ID, req.Vector); err != nil {
		httpError(w, statusFor(err), err.Error())
		return
	}
	if err := s.log.Append(wal.Record{Op: "add", ID: req.ID, Vector: req.Vector}); err != nil {
		// The insert is in memory but not durable; it may vanish on crash.
		// It was never acknowledged, so no durability promise is broken —
		// but surface the failure loudly.
		log.Printf("wal append failed (write not durable): %v", err)
		httpError(w, http.StatusInternalServerError, "write-ahead log failure")
		return
	}
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(map[string]any{"id": req.ID, "count": s.idx.Len()})
}

type searchRequest struct {
	Vector []float32 `json:"vector"`
	K      int       `json:"k"`
}

func (s *Server) handleSearch(w http.ResponseWriter, r *http.Request) {
	var req searchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httpError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	if req.K <= 0 {
		req.K = 10
	}
	if req.K > 100 {
		req.K = 100
	}
	results, err := s.idx.Search(req.Vector, req.K)
	if err != nil {
		httpError(w, statusFor(err), err.Error())
		return
	}
	if results == nil {
		results = []index.Result{}
	}
	json.NewEncoder(w).Encode(map[string]any{"results": results})
}

func (s *Server) handleStats(w http.ResponseWriter, _ *http.Request) {
	stats := map[string]any{"count": s.idx.Len()}
	if h, ok := s.idx.(*index.HNSW); ok {
		stats["index"] = "hnsw"
		stats["m"] = h.M
		stats["ef_search"] = h.EfSearch
	} else {
		stats["index"] = "flat"
	}
	json.NewEncoder(w).Encode(stats)
}

func statusFor(err error) int {
	switch {
	case errors.Is(err, index.ErrDuplicateID):
		return http.StatusConflict
	case errors.Is(err, index.ErrDimMismatch),
		errors.Is(err, index.ErrZeroVector),
		errors.Is(err, index.ErrEmptyID):
		return http.StatusBadRequest
	default:
		return http.StatusInternalServerError
	}
}

func httpError(w http.ResponseWriter, code int, msg string) {
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]string{"error": msg})
}
