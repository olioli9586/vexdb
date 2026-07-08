// Package index provides vector search indexes over unit-normalized vectors.
//
// Similarity is cosine, computed as a dot product because every stored and
// query vector is normalized on the way in. Internally indexes minimize
// distance = 1 - cosine; results expose the cosine score (higher is better).
package index

import (
	"errors"
	"math"
)

// Result is one search hit.
type Result struct {
	ID    string  `json:"id"`
	Score float32 `json:"score"` // cosine similarity, higher is better
}

// Index is a vector search index.
type Index interface {
	Add(id string, vec []float32) error
	Search(vec []float32, k int) ([]Result, error)
	Len() int
}

var (
	ErrDimMismatch = errors.New("vector dimension mismatch")
	ErrZeroVector  = errors.New("vector has zero magnitude")
	ErrDuplicateID = errors.New("id already exists")
	ErrEmptyID     = errors.New("id must not be empty")
)

// Normalize scales v to unit length in place.
func Normalize(v []float32) error {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return ErrZeroVector
	}
	inv := float32(1 / math.Sqrt(sum))
	for i := range v {
		v[i] *= inv
	}
	return nil
}

// dot assumes equal lengths (validated by callers).
func dot(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// distance = 1 - cosine similarity; 0 is identical, 2 is opposite.
func distance(a, b []float32) float32 {
	return 1 - dot(a, b)
}
