package executor

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/registry"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
)

// fableOnlyRejectionHeaders reproduces a capture from api.anthropic.com: the account's
// Fable/overage allowance is exhausted while both shared subscription windows stay allowed,
// so the credential still serves every other model.
func fableOnlyRejectionHeaders() http.Header {
	return http.Header{
		"Content-Type":                                        []string{"application/json"},
		"Anthropic-Ratelimit-Unified-Status":                  []string{"rejected"},
		"Anthropic-Ratelimit-Unified-Representative-Claim":    []string{"seven_day_overage_included"},
		"Anthropic-Ratelimit-Unified-7d_oi-Status":            []string{"rejected"},
		"Anthropic-Ratelimit-Unified-7d_oi-Utilization":       []string{"1.0"},
		"Anthropic-Ratelimit-Unified-7d-Status":               []string{"allowed"},
		"Anthropic-Ratelimit-Unified-7d-Utilization":          []string{"0.72"},
		"Anthropic-Ratelimit-Unified-5h-Status":               []string{"allowed"},
		"Anthropic-Ratelimit-Unified-5h-Utilization":          []string{"0.23"},
		"Anthropic-Ratelimit-Unified-Overage-Status":          []string{"rejected"},
		"Anthropic-Ratelimit-Unified-Overage-Disabled-Reason": []string{"out_of_credits"},
		"Retry-After":                       []string{"17058"},
		"Anthropic-Ratelimit-Unified-Reset": []string{"1790064000"},
		"X-Should-Retry":                    []string{"true"},
	}
}

const fableOnlyRejectionBody = `{"type":"error","error":{"type":"rate_limit_error","message":"This request would exceed your account's rate limit. Please try again later."}}`

func errorIsRequestScoped(err error) bool {
	var scoped cliproxyexecutor.RequestScopedError
	return errors.As(err, &scoped) && scoped != nil && scoped.IsRequestScoped()
}

func errorIsCredentialScoped(err error) bool {
	var scoped interface{ IsCredentialScoped() bool }
	return errors.As(err, &scoped) && scoped != nil && scoped.IsCredentialScoped()
}

// TestClaudeRequestSpeedFastIgnoresBetaDeclaration pins the split between the two meanings of
// fast mode: declaring the beta only permits the speed field, while speed:"fast" is what
// actually selects fast serving and therefore the fast-mode error policy.
func TestClaudeRequestSpeedFastIgnoresBetaDeclaration(t *testing.T) {
	standardBody := []byte(`{"model":"claude-fable-5-1","messages":[{"role":"user","content":"hi"}]}`)
	if claudeRequestSpeedFast(standardBody) {
		t.Fatal("body without a speed field was classified as fast serving")
	}
	if !claudeRequestUsesFastMode(standardBody, map[string]bool{claudeFastModeBeta: true}) {
		t.Fatal("a caller-declared fast-mode beta must still be forwarded on the wire")
	}

	fastBody := []byte(`{"model":"claude-fable-5-1","speed":"fast","messages":[{"role":"user","content":"hi"}]}`)
	if !claudeRequestSpeedFast(fastBody) {
		t.Fatal(`speed:"fast" was not classified as fast serving`)
	}
	if !claudeRequestUsesFastMode(fastBody, nil) {
		t.Fatal(`speed:"fast" must carry the fast-mode beta`)
	}
}

// TestNewClaudeFastDirectResponseErrorScope covers the three shapes a Fast 429 can take.
func TestNewClaudeFastDirectResponseErrorScope(t *testing.T) {
	sharedWindowHeaders := fableOnlyRejectionHeaders()
	sharedWindowHeaders.Set("Anthropic-Ratelimit-Unified-5h-Status", "rejected")

	testCases := []struct {
		name             string
		headers          http.Header
		body             string
		requestScoped    bool
		credentialScoped bool
	}{
		{
			name:    "Fable-only rejection cools the model and rotates",
			headers: fableOnlyRejectionHeaders(),
			body:    fableOnlyRejectionBody,
		},
		{
			name:             "5h rejection still benches the whole credential",
			headers:          sharedWindowHeaders,
			body:             fableOnlyRejectionBody,
			credentialScoped: true,
		},
		{
			name:          "fast-mode entitlement refusal stays with the request",
			headers:       http.Header{"Content-Type": []string{"application/json"}},
			body:          `{"type":"error","error":{"type":"rate_limit_error","message":"Usage credits are required for fast mode"}}`,
			requestScoped: true,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			err := newClaudeFastDirectResponseError(&http.Response{
				StatusCode: http.StatusTooManyRequests,
				Header:     testCase.headers,
				Body:       io.NopCloser(bytes.NewReader(nil)),
			}, []byte(testCase.body))
			if got := errorIsRequestScoped(err); got != testCase.requestScoped {
				t.Fatalf("IsRequestScoped = %v, want %v", got, testCase.requestScoped)
			}
			if got := errorIsCredentialScoped(err); got != testCase.credentialScoped {
				t.Fatalf("IsCredentialScoped = %v, want %v", got, testCase.credentialScoped)
			}
		})
	}
}

// TestClaudeExecutorBetaOnlyRequestUsesStandardErrorPolicy covers the caller that declares
// fast-mode-2026-02-01 without ever sending speed:"fast": Anthropic serves it at the standard
// tier, so a 429 must stay eligible for cooldown and credential rotation instead of inheriting
// the fast-mode request-scoped policy.
func TestClaudeExecutorBetaOnlyRequestUsesStandardErrorPolicy(t *testing.T) {
	var attempts atomic.Int32
	var wireBetas string
	transport := roundTripperFunc(func(req *http.Request) (*http.Response, error) {
		attempts.Add(1)
		requestBody, errRead := io.ReadAll(req.Body)
		if errRead != nil {
			t.Fatal(errRead)
		}
		if bytes.Contains(requestBody, []byte(`"speed"`)) {
			t.Fatalf("beta-only request unexpectedly gained a speed field: %s", requestBody)
		}
		for name, values := range req.Header {
			if strings.EqualFold(name, "Anthropic-Beta") {
				wireBetas = strings.Join(values, ",")
			}
		}
		return &http.Response{
			StatusCode: http.StatusTooManyRequests,
			Header:     fableOnlyRejectionHeaders(),
			Body:       io.NopCloser(strings.NewReader(fableOnlyRejectionBody)),
			Request:    req,
		}, nil
	})

	ctx := context.WithValue(t.Context(), "cliproxy.roundtripper", http.RoundTripper(transport))
	auth := &cliproxyauth.Auth{
		ID:         "beta-only-standard-tier",
		Label:      "beta-only",
		Attributes: map[string]string{"api_key": "sk-ant-oat-beta-only"},
		Metadata:   claudeOAuthTestMetadata(),
	}
	request := cliproxyexecutor.Request{
		Model:   "claude-fable-5-1",
		Payload: []byte(`{"model":"claude-fable-5-1","max_tokens":16,"messages":[{"role":"user","content":"reply OK"}]}`),
	}

	_, errExecute := NewClaudeExecutor(&config.Config{}).Execute(ctx, auth, request, cliproxyexecutor.Options{
		SourceFormat:   sdktranslator.FormatClaude,
		ResponseFormat: sdktranslator.FormatClaude,
		Headers:        http.Header{"Anthropic-Beta": []string{claudeFastModeBeta}},
	})
	if errExecute == nil {
		t.Fatal("expected the rate limit to surface as an error")
	}
	if got := attempts.Load(); got != 1 {
		t.Fatalf("upstream attempts = %d, want 1", got)
	}
	if !strings.Contains(wireBetas, claudeFastModeBeta) {
		t.Fatalf("Anthropic-Beta = %q, want it to keep %s", wireBetas, claudeFastModeBeta)
	}
	var direct *cliproxyexecutor.RequestTerminatedError
	if errors.As(errExecute, &direct) {
		t.Fatal("beta-only request took the Fast direct-response path")
	}
	if errorIsRequestScoped(errExecute) {
		t.Fatal("beta-only rate limit was request-scoped, so no credential state is recorded")
	}
	if errorIsCredentialScoped(errExecute) {
		t.Fatal("Fable-only rate limit benched the whole credential")
	}
}

// TestClaudeExecutorFastFableOnlyRejectionRotates covers a genuine speed:"fast" request meeting
// the same exhausted Fable allowance: still a direct pass-through of the upstream response, but
// scoped to the model so the credential loop tries another account.
func TestClaudeExecutorFastFableOnlyRejectionRotates(t *testing.T) {
	transport := roundTripperFunc(func(req *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusTooManyRequests,
			Header:     fableOnlyRejectionHeaders(),
			Body:       io.NopCloser(strings.NewReader(fableOnlyRejectionBody)),
			Request:    req,
		}, nil
	})
	ctx := context.WithValue(t.Context(), "cliproxy.roundtripper", http.RoundTripper(transport))
	auth := &cliproxyauth.Auth{
		ID:         "fast-fable-only",
		Attributes: map[string]string{"api_key": "sk-ant-oat-fast-fable"},
		Metadata:   claudeOAuthTestMetadata(),
	}
	request := cliproxyexecutor.Request{
		Model:   "claude-fable-5-1",
		Payload: []byte(`{"model":"claude-fable-5-1","max_tokens":16,"speed":"fast","messages":[{"role":"user","content":"reply OK"}]}`),
	}

	_, errExecute := NewClaudeExecutor(&config.Config{}).Execute(ctx, auth, request, cliproxyexecutor.Options{
		SourceFormat:   sdktranslator.FormatClaude,
		ResponseFormat: sdktranslator.FormatClaude,
	})
	var direct *cliproxyexecutor.RequestTerminatedError
	if !errors.As(errExecute, &direct) || direct == nil {
		t.Fatalf("error = %T %v, want the upstream response passed through", errExecute, errExecute)
	}
	if errorIsRequestScoped(errExecute) {
		t.Fatal("Fast Fable-only rate limit was request-scoped, so the credential loop stops")
	}
	if errorIsCredentialScoped(errExecute) {
		t.Fatal("Fast Fable-only rate limit benched the whole credential")
	}
}

// TestClaudeExecutorFableOnly429RotatesToHealthyCredential is the end-to-end contract: a pooled
// account whose Fable allowance is exhausted cools that one model, the request is served by
// another account, and the exhausted account stays available for its other models.
func TestClaudeExecutorFableOnly429RotatesToHealthyCredential(t *testing.T) {
	var exhaustedKey atomic.Value
	var attempts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		apiKey := r.Header.Get("X-Api-Key")
		if apiKey == "" {
			apiKey = strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		}
		if attempts.Add(1) == 1 {
			exhaustedKey.Store(apiKey)
			for name, values := range fableOnlyRejectionHeaders() {
				w.Header().Set(name, values[0])
			}
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(fableOnlyRejectionBody))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"msg-ok","type":"message","model":"claude-fable-5-1","role":"assistant","content":[{"type":"text","text":"ok"}]}`))
	}))
	defer server.Close()

	manager := cliproxyauth.NewManager(nil, nil, nil)
	manager.SetRetryConfig(0, 0, 0)
	manager.RegisterExecutor(NewClaudeExecutor(&config.Config{DisableCooling: false}))

	auths := make([]*cliproxyauth.Auth, 2)
	for index := range auths {
		auths[index] = &cliproxyauth.Auth{
			ID:       uuid.NewString() + "-fable-rotation",
			Provider: "claude",
			Attributes: map[string]string{
				"api_key":  uuid.NewString(),
				"base_url": server.URL,
			},
		}
		registry.GetGlobalRegistry().RegisterClient(auths[index].ID, "claude", []*registry.ModelInfo{{ID: "claude-fable-5-1"}, {ID: "claude-opus-5"}})
		authID := auths[index].ID
		t.Cleanup(func() { registry.GetGlobalRegistry().UnregisterClient(authID) })
		if _, errRegister := manager.Register(context.Background(), auths[index]); errRegister != nil {
			t.Fatalf("register auth: %v", errRegister)
		}
	}

	_, errExecute := manager.Execute(context.Background(), []string{"claude"}, cliproxyexecutor.Request{
		Model:   "claude-fable-5-1",
		Payload: []byte(`{"model":"claude-fable-5-1","messages":[{"role":"user","content":[{"type":"text","text":"test"}]}]}`),
	}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatClaude})
	if errExecute != nil {
		t.Fatalf("expected the second credential to serve the request, got: %v", errExecute)
	}
	if got := attempts.Load(); got != 2 {
		t.Fatalf("upstream attempts = %d, want 2 (rotation to a second credential)", got)
	}

	exhausted := exhaustedKey.Load().(string)
	for _, registered := range auths {
		updated, ok := manager.GetByID(registered.ID)
		if !ok || updated == nil {
			t.Fatal("auth not found")
		}
		state := updated.ModelStates["claude-fable-5-1"]
		if registered.Attributes["api_key"] != exhausted {
			if state != nil && state.Unavailable {
				t.Fatal("the healthy credential was cooled for Fable")
			}
			continue
		}
		// The credential must not be benched as a whole: no credential_quota reason, and no
		// cooldown anywhere near the 17058s (4h44m) reset Anthropic advertised for the Fable
		// allowance. The aggregate deadline only mirrors the one cooled model's short backoff.
		if updated.Quota.Reason == "credential_quota" {
			t.Fatal("Fable-only rejection benched the whole credential")
		}
		if !updated.NextRetryAfter.IsZero() && updated.NextRetryAfter.After(time.Now().Add(time.Minute)) {
			t.Fatalf("credential cooldown = %v, want no long credential-wide bench", updated.NextRetryAfter)
		}
		if state == nil || !state.Unavailable || !state.NextRetryAfter.After(time.Now()) {
			t.Fatalf("Fable model state = %+v, want an active model-level cooldown", state)
		}
		if opus := updated.ModelStates["claude-opus-5"]; opus != nil && opus.Unavailable {
			t.Fatal("the sibling Opus model was cooled on the exhausted credential")
		}
	}
}
