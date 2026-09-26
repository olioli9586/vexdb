package feature

import (
	"hash/fnv"
	"math"
	"testing"
)

// The inlined hash must be exactly FNV-1a: api/demo.snap was built with it,
// so any drift would make every playground query miss its neighbors.
func TestHashIsFNV1a(t *testing.T) {
	for _, s := range []string{"", "a", "^da", "abc", "se$", "ün"} {
		h := fnv.New32a()
		h.Write([]byte(s))
		if got, want := hash(s), h.Sum32(); got != want {
			t.Fatalf("hash(%q) = %#x, want %#x", s, got, want)
		}
	}
}

func TestTrigramNormalizesCaseAndSpace(t *testing.T) {
	a, b := Trigram("Database "), Trigram("database")
	if len(a) != Dims {
		t.Fatalf("len = %d, want %d", len(a), Dims)
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("Trigram differs at %d for case/space variants", i)
		}
	}
}

// Every input, even an empty one, has the "^$" boundary bigram, so the
// vector is never zero and the index never rejects a query.
func TestTrigramNeverZero(t *testing.T) {
	for _, w := range []string{"", " ", "x", "!!"} {
		var sum float32
		for _, x := range Trigram(w) {
			sum += x
		}
		if sum == 0 {
			t.Fatalf("Trigram(%q) is the zero vector", w)
		}
	}
}

func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / math.Sqrt(na*nb)
}

// The point of the featurizer: a typo stays closer to its word than to an
// unrelated word.
func TestTrigramTypoIsCloserThanUnrelated(t *testing.T) {
	for typo, word := range map[string]string{
		"databse":  "database",
		"algoritm": "algorithm",
		"seach":    "search",
	} {
		near := cosine(Trigram(typo), Trigram(word))
		far := cosine(Trigram(typo), Trigram("elephant"))
		if near <= far || near < 0.5 {
			t.Fatalf("%s: cos to %s = %.2f, to elephant = %.2f", typo, word, near, far)
		}
	}
}
