package wal

import (
	"path/filepath"
	"testing"
)

func TestAppendAndReplay(t *testing.T) {
	path := filepath.Join(t.TempDir(), "test.wal")

	w, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	recs := []Record{
		{Op: "add", ID: "a", Vector: []float32{1, 2, 3}},
		{Op: "add", ID: "b", Vector: []float32{4, 5, 6}},
	}
	for _, r := range recs {
		if err := w.Append(r); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	var replayed []Record
	n, err := Replay(path, func(r Record) error {
		replayed = append(replayed, r)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 || len(replayed) != 2 {
		t.Fatalf("replayed %d records, want 2", n)
	}
	if replayed[0].ID != "a" || replayed[1].ID != "b" {
		t.Fatalf("wrong order: %+v", replayed)
	}
	if replayed[1].Vector[2] != 6 {
		t.Fatalf("vector corrupted: %+v", replayed[1])
	}
}

func TestReplayMissingFileIsEmpty(t *testing.T) {
	n, err := Replay(filepath.Join(t.TempDir(), "nope.wal"), func(Record) error { return nil })
	if err != nil || n != 0 {
		t.Fatalf("missing file should be empty db, got n=%d err=%v", n, err)
	}
}
