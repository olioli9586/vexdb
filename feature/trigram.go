// Package feature turns strings into vectors for the demo playground.
//
// Character-trigram hashing: pad the word, slide a 3-char window, hash each
// trigram into one of Dims buckets, count, normalize. Words sharing trigrams
// land near each other in cosine space — which makes nearest-neighbor search
// behave like typo-tolerant fuzzy matching. No ML model needed, fully
// deterministic, and an honest way to demo a vector index on real strings.
package feature

import "strings"

const Dims = 128

func Trigram(word string) []float32 {
	w := "^" + strings.ToLower(strings.TrimSpace(word)) + "$"
	v := make([]float32, Dims)
	// Bigrams + trigrams: bigrams give partial credit when a typo breaks a
	// trigram, which noticeably improves fuzzy matching.
	for i := 0; i+2 <= len(w); i++ {
		v[hash(w[i:i+2])%Dims]++
	}
	for i := 0; i+3 <= len(w); i++ {
		v[hash(w[i:i+3])%Dims]++
	}
	return v
}

// FNV-1a, inlined to keep the package dependency-free.
func hash(s string) uint32 {
	h := uint32(2166136261)
	for i := 0; i < len(s); i++ {
		h ^= uint32(s[i])
		h *= 16777619
	}
	return h
}
