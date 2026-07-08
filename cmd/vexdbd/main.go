// vexdbd runs the VexDB server: replay the WAL into a fresh HNSW index,
// then serve HTTP.
package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"

	"github.com/olioli9586/vexdb/index"
	"github.com/olioli9586/vexdb/server"
	"github.com/olioli9586/vexdb/wal"
)

func main() {
	addr := flag.String("addr", ":8080", "listen address")
	walPath := flag.String("wal", "vexdb.wal", "write-ahead log path")
	m := flag.Int("m", 16, "HNSW M (links per node per layer)")
	efc := flag.Int("ef-construction", 200, "HNSW construction beam width")
	efs := flag.Int("ef-search", 64, "HNSW search beam width")
	flag.Parse()

	idx := index.NewHNSW(*m, *efc, *efs)
	skipped := 0
	n, err := wal.Replay(*walPath, func(rec wal.Record) error {
		if addErr := idx.Add(rec.ID, rec.Vector); addErr != nil {
			// A validation failure in an old log (e.g. written by a version
			// that logged before validating) shouldn't brick startup — warn
			// and continue. Corrupt JSON still fails hard in Replay itself.
			log.Printf("wal replay: skipping record id=%q: %v", rec.ID, addErr)
			skipped++
		}
		return nil
	})
	if err != nil {
		log.Fatalf("wal replay: %v", err)
	}
	fmt.Printf("vexdbd: replayed %d records (%d skipped) from %s\n", n-skipped, skipped, *walPath)

	w, err := wal.Open(*walPath)
	if err != nil {
		log.Fatalf("wal open: %v", err)
	}
	defer w.Close()

	fmt.Printf("vexdbd: listening on %s (HNSW M=%d efC=%d efS=%d)\n", *addr, *m, *efc, *efs)
	log.Fatal(http.ListenAndServe(*addr, server.New(idx, w)))
}
