// Package wal is a write-ahead log: every accepted insert is appended to an
// on-disk log before it is acknowledged, so a crash never loses acknowledged
// data. On startup the log is replayed to rebuild the in-memory index.
//
// Format: one JSON record per line. JSONL trades some space for a log you
// can inspect with `tail` and `jq` — the right call until compaction/
// snapshotting (roadmap) makes binary encoding worth it.
package wal

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sync"
)

type Record struct {
	Op     string    `json:"op"` // "add"
	ID     string    `json:"id"`
	Vector []float32 `json:"vector"`
}

type WAL struct {
	mu   sync.Mutex
	f    *os.File
	w    *bufio.Writer
	path string
}

func Open(path string) (*WAL, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, err
	}
	return &WAL{f: f, w: bufio.NewWriter(f), path: path}, nil
}

// Append durably writes one record (flush + fsync) before returning.
func (w *WAL) Append(rec Record) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	b, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	if _, err := w.w.Write(append(b, '\n')); err != nil {
		return err
	}
	if err := w.w.Flush(); err != nil {
		return err
	}
	return w.f.Sync()
}

// Replay streams every record in the log through fn, in write order.
func Replay(path string, fn func(Record) error) (int, error) {
	f, err := os.Open(path)
	if os.IsNotExist(err) {
		return 0, nil // fresh database
	}
	if err != nil {
		return 0, err
	}
	defer f.Close()

	count := 0
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<24) // vectors can make long lines
	for sc.Scan() {
		var rec Record
		if err := json.Unmarshal(sc.Bytes(), &rec); err != nil {
			return count, fmt.Errorf("wal line %d corrupt: %w", count+1, err)
		}
		if err := fn(rec); err != nil {
			return count, fmt.Errorf("wal replay line %d: %w", count+1, err)
		}
		count++
	}
	return count, sc.Err()
}

func (w *WAL) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if err := w.w.Flush(); err != nil {
		return err
	}
	return w.f.Close()
}
