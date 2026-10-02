package main

// Concurrency regression tests for the LSP client.
//
// Each test pins down a correctness bug by its observable behavior, using only
// APIs that exist before and after the fix:
//   - TestEnsureOpenSendsSingleDidOpen: concurrent ensureOpen must not send
//     textDocument/didOpen twice for the same URI (check-then-act race).
//   - TestSyncDocVersionsIncreaseMonotonically: concurrent syncDoc must not
//     emit duplicate document versions (lost-update race on the version
//     counter; LSP requires monotonically increasing versions).
//   - TestWriteDoesNotHoldMutexDuringPipeIO: a blocked pipe write must not
//     wedge unrelated client operations (alive must stay responsive).
//   - TestCallCleansPendingOnMarshalFailure / TestCallCleansPendingOnWriteFailure:
//     a request that never reaches the server must not leak its pending slot.
//
// A fake language server on an io.Pipe pair records what the client sends, so
// no real server binary is needed.

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// lspRaceFakeServer records the notifications a client sends.
type lspRaceFakeServer struct {
	mu            sync.Mutex
	didOpen       int
	didChangeVers []int
}

func (s *lspRaceFakeServer) run(r *io.PipeReader) {
	br := bufio.NewReader(r)
	for {
		msg, err := readFrame(br)
		if err != nil {
			return
		}
		if len(msg.ID) > 0 {
			continue // the tests below only send notifications
		}
		s.mu.Lock()
		switch msg.Method {
		case "textDocument/didOpen":
			s.didOpen++
		case "textDocument/didChange":
			var p struct {
				TextDocument struct {
					Version int `json:"version"`
				} `json:"textDocument"`
			}
			if json.Unmarshal(msg.Params, &p) == nil {
				s.didChangeVers = append(s.didChangeVers, p.TextDocument.Version)
			}
		}
		s.mu.Unlock()
	}
}

// lspRaceWireClient builds a client whose pipes are looped to a fake server
// instead of a spawned process. On the fixed implementation it also starts the
// dedicated writer loop (via a capability check, so this still compiles and
// runs against the old code, where writes are synchronous). It takes
// testing.TB so benchmarks can reuse it too.
func lspRaceWireClient(t testing.TB) (*lspClient, *lspRaceFakeServer) {
	t.Helper()
	cl := newLSPClient(lspServerDef{Name: "fake", Cmd: []string{"fake"}}, t.TempDir())
	toSrvR, toSrvW := io.Pipe() // client -> server
	toCliR, toCliW := io.Pipe() // server -> client (never written in these tests)
	cl.in = toSrvW
	cl.out = bufio.NewReader(toCliR)
	srv := &lspRaceFakeServer{}
	go srv.run(toSrvR)
	go cl.readLoop()
	if starter, ok := any(cl).(interface{ beginWriteLoop() }); ok {
		starter.beginWriteLoop()
	}
	t.Cleanup(func() {
		toSrvR.Close()
		toSrvW.Close()
		toCliR.Close()
		toCliW.Close()
	})
	return cl, srv
}

func TestEnsureOpenSendsSingleDidOpen(t *testing.T) {
	for round := 0; round < 3; round++ {
		cl, srv := lspRaceWireClient(t)
		path := filepath.Join(t.TempDir(), "f.go")
		if err := os.WriteFile(path, []byte("package main\n"), 0644); err != nil {
			t.Fatal(err)
		}
		const n = 32
		var wg sync.WaitGroup
		for i := 0; i < n; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				_ = cl.ensureOpen(path, "f.go")
			}()
		}
		wg.Wait()
		time.Sleep(500 * time.Millisecond) // let the fake server drain the pipe
		srv.mu.Lock()
		got := srv.didOpen
		srv.mu.Unlock()
		if got != 1 {
			t.Fatalf("round %d: server got %d textDocument/didOpen, want exactly 1", round, got)
		}
	}
}

func TestSyncDocVersionsIncreaseMonotonically(t *testing.T) {
	for round := 0; round < 3; round++ {
		cl, srv := lspRaceWireClient(t)
		path := filepath.Join(t.TempDir(), "f.go")
		if err := os.WriteFile(path, []byte("package main\n"), 0644); err != nil {
			t.Fatal(err)
		}
		if err := cl.ensureOpen(path, "f.go"); err != nil {
			t.Fatalf("ensureOpen: %v", err)
		}
		// One real change, then a storm of concurrent syncs over identical
		// content: every didChange the server sees must carry a distinct,
		// gap-free version starting at 2.
		if err := os.WriteFile(path, []byte("package main\n\n// changed\n"), 0644); err != nil {
			t.Fatal(err)
		}
		const n = 64
		var wg sync.WaitGroup
		for i := 0; i < n; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				_ = cl.syncDoc(path, "f.go")
			}()
		}
		wg.Wait()
		time.Sleep(500 * time.Millisecond) // let the fake server drain the pipe
		srv.mu.Lock()
		vers := append([]int(nil), srv.didChangeVers...)
		srv.mu.Unlock()

		seen := map[int]bool{}
		for _, v := range vers {
			if seen[v] {
				t.Fatalf("round %d: duplicate didChange version %d in %v", round, v, vers)
			}
			seen[v] = true
		}
		if len(vers) == 0 {
			t.Fatalf("round %d: no didChange sent for a real content change", round)
		}
		min, max := vers[0], vers[0]
		for _, v := range vers[1:] {
			if v < min {
				min = v
			}
			if v > max {
				max = v
			}
		}
		if min != 2 || max-min+1 != len(vers) {
			t.Fatalf("round %d: versions %v are not contiguous starting at 2", round, vers)
		}
	}
}

func TestWriteDoesNotHoldMutexDuringPipeIO(t *testing.T) {
	cl := newLSPClient(lspServerDef{Name: "fake", Cmd: []string{"fake"}}, t.TempDir())
	r, w := io.Pipe() // never drained: writes block once the buffer fills
	cl.in = w
	if starter, ok := any(cl).(interface{ beginWriteLoop() }); ok {
		starter.beginWriteLoop()
	}
	t.Cleanup(func() { w.Close(); r.Close() })

	big := strings.Repeat("x", 1<<20) // 1MB: far beyond any pipe buffer
	done := make(chan error, 1)
	go func() {
		done <- cl.notify("textDocument/didOpen", map[string]any{
			"textDocument": map[string]any{"uri": "file:///big.go", "text": big},
		})
	}()
	time.Sleep(300 * time.Millisecond) // let the write wedge itself in the pipe

	aliveCh := make(chan error, 1)
	go func() { aliveCh <- cl.alive() }()
	select {
	case err := <-aliveCh:
		if err != nil {
			t.Fatalf("alive() = %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("alive() blocked for 2s: write() holds the client mutex during pipe I/O")
	}

	w.Close() // release the wedged writer, whichever implementation it is
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("notify never returned after the pipe was closed")
	}
}

func TestCallCleansPendingOnMarshalFailure(t *testing.T) {
	cl := newLSPClient(lspServerDef{Name: "fake", Cmd: []string{"fake"}}, t.TempDir())
	// A func value cannot be marshalled to JSON.
	err := cl.call(context.Background(), "test/method", func() {}, nil)
	if err == nil {
		t.Fatal("expected json.Marshal to fail on a func param")
	}
	cl.mu.Lock()
	n := len(cl.pending)
	cl.mu.Unlock()
	if n != 0 {
		t.Fatalf("pending holds %d entries after marshal failure: leaked", n)
	}
}

// lspRaceFailWriter is an io.WriteCloser whose writes always fail.
type lspRaceFailWriter struct{}

func (lspRaceFailWriter) Write([]byte) (int, error) { return 0, errors.New("boom") }
func (lspRaceFailWriter) Close() error              { return nil }

func TestCallCleansPendingOnWriteFailure(t *testing.T) {
	cl := newLSPClient(lspServerDef{Name: "fake", Cmd: []string{"fake"}}, t.TempDir())
	cl.in = lspRaceFailWriter{}
	if starter, ok := any(cl).(interface{ beginWriteLoop() }); ok {
		starter.beginWriteLoop()
	}
	err := cl.call(context.Background(), "test/method", map[string]any{"a": 1}, nil)
	if err == nil {
		t.Fatal("expected the write to fail")
	}
	cl.mu.Lock()
	n := len(cl.pending)
	cl.mu.Unlock()
	if n != 0 {
		t.Fatalf("pending holds %d entries after write failure: leaked", n)
	}
}
