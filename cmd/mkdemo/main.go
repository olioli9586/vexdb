// mkdemo builds the playground snapshot: a sample of dictionary words
// embedded as character-trigram vectors and indexed with HNSW, serialized
// with the snapshot format so the serverless demo loads it in ~100ms
// instead of rebuilding the graph on every cold start.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/olioli9586/vexdb/feature"
	"github.com/olioli9586/vexdb/index"
)

func main() {
	common := flag.String("common", "/tmp/g10k.txt", "frequency-ranked common words (indexed first, all of them)")
	dict := flag.String("dict", "/usr/share/dict/words", "full dictionary (sampled for volume)")
	out := flag.String("out", "api/demo.snap", "output snapshot path")
	every := flag.Int("every", 12, "keep every Nth eligible dictionary word")
	flag.Parse()

	idx := index.NewHNSW(12, 120, 48)
	kept := 0
	add := func(w string) {
		if err := idx.Add(w, feature.Trigram(w)); err == nil {
			kept++
		}
	}

	// All common words first (so the things people actually type are present),
	// then a uniform dictionary sample for volume.
	if f, err := os.Open(*common); err == nil {
		sc := bufio.NewScanner(f)
		for sc.Scan() {
			if w := clean(sc.Text()); len(w) >= 3 {
				add(w)
			}
		}
		f.Close()
	}
	f, err := os.Open(*dict)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	seen := 0
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		w := clean(sc.Text())
		if len(w) < 4 || len(w) > 12 {
			continue
		}
		seen++
		if seen%*every != 0 {
			continue
		}
		add(w)
	}
	if err := sc.Err(); err != nil {
		panic(err)
	}

	o, err := os.Create(*out)
	if err != nil {
		panic(err)
	}
	defer o.Close()
	if err := idx.Save(o); err != nil {
		panic(err)
	}
	st, _ := o.Stat()
	fmt.Printf("mkdemo: indexed %d words -> %s (%.1f MB)\n", kept, *out, float64(st.Size())/1e6)
}

// clean lowercases and rejects anything non-alphabetic.
func clean(s string) string {
	w := strings.ToLower(strings.TrimSpace(s))
	for _, r := range w {
		if r < 'a' || r > 'z' {
			return ""
		}
	}
	return w
}
