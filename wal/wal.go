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
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sync"
)

// maxRecordBytes bounds one encoded record including its newline. Replay's
// line buffer is this size, and Append refuses anything larger, so every
// acknowledged record can be replayed.
const maxRecordBytes = 1 << 24

var ErrRecordTooLarge = errors.New("wal: record exceeds maximum size")

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
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR|os.O_APPEND, 0o644)
	if err != nil {
		return nil, err
	}
	if err := repairTail(f); err != nil {
		f.Close()
		return nil, fmt.Errorf("wal: repairing log tail: %w", err)
	}
	return &WAL{f: f, w: bufio.NewWriter(f), path: path}, nil
}

// repairTail makes the log end on a record boundary before new appends. A
// crash during Append can leave a final line with no newline. If that line
// is a complete record it is kept (Replay applies it too) and terminated;
// otherwise it is a torn write, never acknowledged, and is truncated. Either
// way the next record starts on its own line instead of being glued onto
// the fragment, which would corrupt it.
func repairTail(f *os.File) error {
	info, err := f.Stat()
	if err != nil {
		return err
	}
	size := info.Size()
	if size == 0 {
		return nil
	}
	var last [1]byte
	if _, err := f.ReadAt(last[:], size-1); err != nil {
		return err
	}
	if last[0] == '\n' {
		return nil // clean shutdown or completed append: the common case
	}

	// The unterminated tail starts just past the last newline.
	start := int64(0)
	buf := make([]byte, 64<<10)
	for end := size; end > 0; {
		off := max(0, end-int64(len(buf)))
		chunk := buf[:end-off]
		if _, err := f.ReadAt(chunk, off); err != nil {
			return err
		}
		if i := bytes.LastIndexByte(chunk, '\n'); i >= 0 {
			start = off + int64(i) + 1
			break
		}
		end = off
	}

	complete := false
	if size-start < maxRecordBytes {
		tail := make([]byte, size-start)
		if _, err := f.ReadAt(tail, start); err != nil {
			return err
		}
		var rec Record
		complete = json.Unmarshal(tail, &rec) == nil
	}
	if complete {
		_, err = f.Write([]byte{'\n'})
	} else {
		err = f.Truncate(start)
	}
	if err != nil {
		return err
	}
	return f.Sync()
}

// Append durably writes one record (flush + fsync) before returning.
func (w *WAL) Append(rec Record) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	b, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	if len(b)+1 > maxRecordBytes {
		return fmt.Errorf("%w (%d bytes, limit %d)", ErrRecordTooLarge, len(b)+1, maxRecordBytes)
	}
	if _, err := w.w.Write(append(b, '\n')); err != nil {
		return err
	}
	if err := w.w.Flush(); err != nil {
		return err
	}
	return w.f.Sync()
}

// Replay streams every record in the log through fn, in write order. An
// unparseable final line with no newline is a write torn by a crash and is
// skipped (Open truncates it); a bad line anywhere else is an error.
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
	sc.Buffer(make([]byte, 1<<20), maxRecordBytes) // vectors can make long lines
	unterminated := false
	sc.Split(func(data []byte, atEOF bool) (int, []byte, error) {
		if atEOF && len(data) > 0 && bytes.IndexByte(data, '\n') < 0 {
			unterminated = true // this is the last line and it has no newline
		}
		return bufio.ScanLines(data, atEOF)
	})
	for sc.Scan() {
		var rec Record
		if err := json.Unmarshal(sc.Bytes(), &rec); err != nil {
			if unterminated {
				// Torn by a crash mid-Append. Append returns only after
				// the newline is synced, so it was never acknowledged.
				break
			}
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
