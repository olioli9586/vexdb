package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/olioli9586/vexdb/index"
	"github.com/olioli9586/vexdb/wal"
)

type harness struct {
	t       *testing.T
	srv     *Server
	walPath string
}

func newHarness(t *testing.T, idx index.Index) *harness {
	t.Helper()
	path := filepath.Join(t.TempDir(), "test.wal")
	w, err := wal.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { w.Close() })
	return &harness{t: t, srv: New(idx, w), walPath: path}
}

func (h *harness) do(method, path, body string) (int, map[string]any) {
	h.t.Helper()
	rec := httptest.NewRecorder()
	h.srv.ServeHTTP(rec, httptest.NewRequest(method, path, strings.NewReader(body)))
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		h.t.Fatalf("%s %s: content-type %q, want application/json", method, path, ct)
	}
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		h.t.Fatalf("%s %s: response is not JSON: %v (%q)", method, path, err, rec.Body.String())
	}
	return rec.Code, out
}

func (h *harness) logged() []wal.Record {
	h.t.Helper()
	var recs []wal.Record
	if _, err := wal.Replay(h.walPath, func(r wal.Record) error {
		recs = append(recs, r)
		return nil
	}); err != nil {
		h.t.Fatal(err)
	}
	return recs
}

func resultIDs(t *testing.T, body map[string]any) []string {
	t.Helper()
	raw, ok := body["results"].([]any)
	if !ok {
		t.Fatalf("results is %T, want an array: %v", body["results"], body)
	}
	ids := make([]string, len(raw))
	for i, r := range raw {
		ids[i] = r.(map[string]any)["id"].(string)
	}
	return ids
}

func TestAddSearchStats(t *testing.T) {
	h := newHarness(t, index.NewHNSW(16, 200, 64))
	for i, body := range []string{
		`{"id":"east","vector":[1,0]}`,
		`{"id":"north","vector":[0,1]}`,
		`{"id":"northeast","vector":[3,3]}`,
	} {
		code, out := h.do("POST", "/vectors", body)
		if code != http.StatusCreated {
			t.Fatalf("add %d: status %d: %v", i, code, out)
		}
		if out["count"] != float64(i+1) {
			t.Fatalf("add %d: count %v", i, out["count"])
		}
	}

	code, out := h.do("POST", "/search", `{"vector":[2,0.1],"k":2}`)
	if code != http.StatusOK {
		t.Fatalf("search: status %d: %v", code, out)
	}
	if got := strings.Join(resultIDs(t, out), ","); got != "east,northeast" {
		t.Fatalf("search: got %s, want east,northeast", got)
	}

	code, out = h.do("GET", "/stats", "")
	if code != http.StatusOK || out["count"] != float64(3) || out["index"] != "hnsw" || out["m"] != float64(16) {
		t.Fatalf("stats: %d %v", code, out)
	}

	// Every acknowledged write is in the log, with the vector as sent
	// (normalization happens in the index, not in the log).
	recs := h.logged()
	if len(recs) != 3 || recs[2].ID != "northeast" || recs[2].Vector[0] != 3 {
		t.Fatalf("wal: %+v", recs)
	}
}

// Rejected writes must not reach the log: a logged duplicate once made
// replay fail at startup.
func TestRejectedAddsAreNotLogged(t *testing.T) {
	h := newHarness(t, index.NewFlat())
	if code, out := h.do("POST", "/vectors", `{"id":"a","vector":[1,2]}`); code != http.StatusCreated {
		t.Fatalf("status %d: %v", code, out)
	}
	cases := []struct {
		body string
		want int
	}{
		{`{"id":"a","vector":[3,4]}`, http.StatusConflict},
		{`{"id":"","vector":[1,2]}`, http.StatusBadRequest},
		{`{"id":"z","vector":[0,0]}`, http.StatusBadRequest},
		{`{"id":"d","vector":[1,2,3]}`, http.StatusBadRequest},
		{`{"id":"j","vector":[1,2]`, http.StatusBadRequest},
		{`{"id":"s","vector":"nope"}`, http.StatusBadRequest},
	}
	for _, c := range cases {
		code, out := h.do("POST", "/vectors", c.body)
		if code != c.want || out["error"] == nil {
			t.Fatalf("%s: status %d (%v), want %d with an error", c.body, code, out, c.want)
		}
	}
	if recs := h.logged(); len(recs) != 1 || recs[0].ID != "a" {
		t.Fatalf("wal should hold only the accepted write, got %+v", recs)
	}
}

func TestSearchEmptyIndexReturnsEmptyArray(t *testing.T) {
	h := newHarness(t, index.NewHNSW(16, 200, 64))
	code, out := h.do("POST", "/search", `{"vector":[1,0]}`)
	if code != http.StatusOK {
		t.Fatalf("status %d: %v", code, out)
	}
	if ids := resultIDs(t, out); len(ids) != 0 {
		t.Fatalf("got %v", ids)
	}
}

func TestSearchErrors(t *testing.T) {
	h := newHarness(t, index.NewFlat())
	h.do("POST", "/vectors", `{"id":"a","vector":[1,2]}`)
	for body, want := range map[string]int{
		`{"vector":[1,2,3]}`: http.StatusBadRequest,
		`{"vector":[0,0]}`:   http.StatusBadRequest,
		`not json`:           http.StatusBadRequest,
	} {
		if code, out := h.do("POST", "/search", body); code != want || out["error"] == nil {
			t.Fatalf("%s: status %d (%v), want %d", body, code, out, want)
		}
	}
}

func TestSearchClampsK(t *testing.T) {
	h := newHarness(t, index.NewFlat())
	for i := 0; i < 120; i++ {
		body, _ := json.Marshal(map[string]any{"id": string(rune('A'+i%26)) + strings.Repeat("x", i/26), "vector": []float32{float32(i + 1), 1}})
		if code, out := h.do("POST", "/vectors", string(body)); code != http.StatusCreated {
			t.Fatalf("add %d: %d %v", i, code, out)
		}
	}
	for body, want := range map[string]int{
		`{"vector":[1,1]}`:          10,
		`{"vector":[1,1],"k":-5}`:   10,
		`{"vector":[1,1],"k":3}`:    3,
		`{"vector":[1,1],"k":5000}`: 100,
	} {
		_, out := h.do("POST", "/search", body)
		if got := len(resultIDs(t, out)); got != want {
			t.Fatalf("%s: %d results, want %d", body, got, want)
		}
	}
}

// Subnormal components are valid JSON floats. They used to normalize to
// [+Inf, NaN], after which encoding any search result containing that
// vector failed and the client got an empty 200 response.
func TestSubnormalVectorIsSearchable(t *testing.T) {
	h := newHarness(t, index.NewHNSW(16, 200, 64))
	for _, body := range []string{`{"id":"tiny","vector":[1e-45,0]}`, `{"id":"north","vector":[0,1]}`} {
		if code, out := h.do("POST", "/vectors", body); code != http.StatusCreated {
			t.Fatalf("status %d: %v", code, out)
		}
	}
	code, out := h.do("POST", "/search", `{"vector":[1,0],"k":2}`)
	if code != http.StatusOK {
		t.Fatalf("status %d: %v", code, out)
	}
	if ids := resultIDs(t, out); len(ids) != 2 || ids[0] != "tiny" {
		t.Fatalf("got %v, want tiny first", ids)
	}
}

func TestOversizedBodyRejected(t *testing.T) {
	h := newHarness(t, index.NewFlat())
	vec := "[" + strings.Repeat("1,", maxBodyBytes/2) + "1]"
	for _, path := range []string{"/vectors", "/search"} {
		code, out := h.do("POST", path, `{"id":"big","vector":`+vec+`}`)
		if code != http.StatusRequestEntityTooLarge {
			t.Fatalf("%s: status %d (%v), want 413", path, code, out)
		}
	}
	if recs := h.logged(); len(recs) != 0 {
		t.Fatalf("oversized write reached the wal: %d records", len(recs))
	}
}

func TestStatusFor(t *testing.T) {
	for err, want := range map[error]int{
		index.ErrDuplicateID:  http.StatusConflict,
		index.ErrDimMismatch:  http.StatusBadRequest,
		index.ErrZeroVector:   http.StatusBadRequest,
		index.ErrNonFinite:    http.StatusBadRequest,
		index.ErrEmptyID:      http.StatusBadRequest,
		wal.ErrRecordTooLarge: http.StatusInternalServerError,
	} {
		if got := statusFor(err); got != want {
			t.Fatalf("statusFor(%v) = %d, want %d", err, got, want)
		}
	}
}
