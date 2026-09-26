package wal

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
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

func appendRecords(t *testing.T, path string, recs ...Record) {
	t.Helper()
	w, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range recs {
		if err := w.Append(r); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
}

func appendRaw(t *testing.T, path, s string) {
	t.Helper()
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString(s); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

func replayIDs(t *testing.T, path string) ([]string, error) {
	t.Helper()
	var ids []string
	n, err := Replay(path, func(r Record) error {
		ids = append(ids, r.ID)
		return nil
	})
	if n != len(ids) {
		t.Fatalf("Replay returned count %d but applied %d records", n, len(ids))
	}
	return ids, err
}

func wantIDs(t *testing.T, got []string, err error, want ...string) {
	t.Helper()
	if err != nil {
		t.Fatalf("replay: %v", err)
	}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("replayed %v, want %v", got, want)
	}
}

var (
	recA = Record{Op: "add", ID: "a", Vector: []float32{1, 0}}
	recB = Record{Op: "add", ID: "b", Vector: []float32{0, 1}}
	recC = Record{Op: "add", ID: "c", Vector: []float32{1, 1}}
)

// A crash in the middle of Append leaves a partial last line. Append only
// returns after the newline is fsync'd, so that record was never
// acknowledged: replay must drop it rather than refuse to start.
func TestReplayIgnoresTornTail(t *testing.T) {
	path := filepath.Join(t.TempDir(), "torn.wal")
	appendRecords(t, path, recA, recB)
	appendRaw(t, path, `{"op":"add","id":"c","vec`)

	ids, err := replayIDs(t, path)
	wantIDs(t, ids, err, "a", "b")
}

// Open must cut the torn fragment off. Otherwise the next (acknowledged)
// record is appended onto the same line, which then fails to parse and
// takes that record down with it.
func TestOpenRepairsTornTail(t *testing.T) {
	path := filepath.Join(t.TempDir(), "torn.wal")
	appendRecords(t, path, recA, recB)
	appendRaw(t, path, `{"op":"add","id":"x","vec`)
	appendRecords(t, path, recC)

	ids, err := replayIDs(t, path)
	wantIDs(t, ids, err, "a", "b", "c")
}

// The fragment can be longer than Open's backwards read buffer, and the
// log can consist of nothing but a fragment.
func TestOpenRepairsLongTornTailWithoutNewline(t *testing.T) {
	path := filepath.Join(t.TempDir(), "torn.wal")
	appendRaw(t, path, `{"op":"add","id":"x","vector":[`+strings.Repeat("0.5,", 50_000))
	appendRecords(t, path, recA)
	ids, err := replayIDs(t, path)
	wantIDs(t, ids, err, "a")

	path = filepath.Join(t.TempDir(), "torn2.wal")
	appendRecords(t, path, recA)
	appendRaw(t, path, `{"op":"add","id":"x","vector":[`+strings.Repeat("0.5,", 50_000))
	appendRecords(t, path, recB)
	ids, err = replayIDs(t, path)
	wantIDs(t, ids, err, "a", "b")
}

// A complete record that is only missing its newline is kept by Replay,
// and Open terminates it so the next record starts on its own line.
func TestUnterminatedCompleteRecordIsKept(t *testing.T) {
	path := filepath.Join(t.TempDir(), "noeol.wal")
	appendRecords(t, path, recA)
	appendRaw(t, path, `{"op":"add","id":"b","vector":[0,1]}`)

	ids, err := replayIDs(t, path)
	wantIDs(t, ids, err, "a", "b")

	appendRecords(t, path, recC)
	ids, err = replayIDs(t, path)
	wantIDs(t, ids, err, "a", "b", "c")
}

// Only the unterminated final line gets the torn-write treatment; a bad
// line in the middle of the log is real corruption and must fail loudly.
func TestReplayCorruptMiddleLineFails(t *testing.T) {
	path := filepath.Join(t.TempDir(), "corrupt.wal")
	appendRecords(t, path, recA)
	appendRaw(t, path, "not json\n")
	appendRecords(t, path, recB)

	if _, err := replayIDs(t, path); err == nil || !strings.Contains(err.Error(), "line 2") {
		t.Fatalf("want a line-2 corruption error, got %v", err)
	}
}

// Replay's line buffer is capped at maxRecordBytes, so Append must refuse
// anything larger: an acknowledged record that replay cannot read would
// stop the server from starting. A record of exactly the limit must work.
func TestAppendRecordSizeLimit(t *testing.T) {
	path := filepath.Join(t.TempDir(), "big.wal")
	empty, err := json.Marshal(Record{Op: "add"})
	if err != nil {
		t.Fatal(err)
	}
	fits := Record{Op: "add", ID: strings.Repeat("x", maxRecordBytes-1-len(empty))}
	tooBig := Record{Op: "add", ID: fits.ID + "x"}

	w, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.Append(tooBig); !errors.Is(err, ErrRecordTooLarge) {
		t.Fatalf("want ErrRecordTooLarge, got %v", err)
	}
	for _, r := range []Record{recA, fits, recB} {
		if err := w.Append(r); err != nil {
			t.Fatalf("append %d-byte id: %v", len(r.ID), err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	ids, err := replayIDs(t, path)
	if err != nil {
		t.Fatalf("replay: %v", err)
	}
	if len(ids) != 3 || ids[0] != "a" || len(ids[1]) != len(fits.ID) || ids[2] != "b" {
		t.Fatalf("replayed %d records", len(ids))
	}
}
