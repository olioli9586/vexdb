package handler

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/olioli9586/vexdb/index"
)

type searchResponse struct {
	Query     string         `json:"query"`
	Results   []index.Result `json:"results"`
	Indexed   int            `json:"indexed"`
	LatencyUS int64          `json:"latency_us"`
	Error     string         `json:"error"`
}

func get(t *testing.T, q string) (*httptest.ResponseRecorder, searchResponse) {
	t.Helper()
	rec := httptest.NewRecorder()
	Handler(rec, httptest.NewRequest(http.MethodGet, "/api/search?q="+url.QueryEscape(q), nil))
	var body searchResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("q=%q: response is not JSON: %v (%q)", q, err, rec.Body.String())
	}
	return rec, body
}

// The embedded snapshot must load (this also checks it against LoadHNSW's
// structural validation) and typo'd queries must find the intended word.
func TestHandlerFindsTypoTarget(t *testing.T) {
	for q, want := range map[string]string{
		"databse": "database",
		"seach":   "search",
		"vektor":  "vector",
	} {
		rec, body := get(t, q)
		if rec.Code != http.StatusOK {
			t.Fatalf("q=%q: status %d: %s", q, rec.Code, body.Error)
		}
		if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
			t.Fatalf("content-type %q", ct)
		}
		if rec.Header().Get("Access-Control-Allow-Origin") != "*" {
			t.Fatal("missing CORS header")
		}
		if body.Indexed == 0 || len(body.Results) != 10 {
			t.Fatalf("q=%q: indexed=%d, %d results", q, body.Indexed, len(body.Results))
		}
		found := false
		for _, r := range body.Results {
			if r.ID == want {
				found = true
			}
		}
		if !found {
			t.Fatalf("q=%q: %q not in results %v", q, want, body.Results)
		}
		for i := 1; i < len(body.Results); i++ {
			if body.Results[i].Score > body.Results[i-1].Score {
				t.Fatalf("q=%q: results not sorted by score: %v", q, body.Results)
			}
		}
	}
}

func TestHandlerRejectsBadQueries(t *testing.T) {
	for _, q := range []string{"", "   ", strings.Repeat("a", 41)} {
		rec, body := get(t, q)
		if rec.Code != http.StatusBadRequest || body.Error == "" {
			t.Fatalf("q=%q: status %d, error %q; want 400 with an error", q, rec.Code, body.Error)
		}
	}
	// 40 characters is the documented maximum and must still be served.
	if rec, _ := get(t, strings.Repeat("a", 40)); rec.Code != http.StatusOK {
		t.Fatalf("40-char query: status %d, want 200", rec.Code)
	}
}
