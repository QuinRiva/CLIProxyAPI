package executor

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	log "github.com/sirupsen/logrus"

	"github.com/router-for-me/CLIProxyAPI/v8/internal/runtime/executor/helps"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/executor"
)

// claudeFastRequestError marks a Fast request failure as request-scoped. Fast
// errors must stop at the caller: they do not justify retrying another
// credential or changing the selected credential's availability, unless the failure
// is a genuine credential-level rate limit.
type claudeFastRequestError struct {
	cause      error
	status     int
	retryAfter *time.Duration
}

func (e *claudeFastRequestError) Error() string {
	if e == nil || e.cause == nil {
		return ""
	}
	return e.cause.Error()
}

func (e *claudeFastRequestError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.cause
}

func (e *claudeFastRequestError) StatusCode() int {
	if e == nil {
		return 0
	}
	if e.status <= 0 || (e.status >= http.StatusOK && e.status < http.StatusMultipleChoices) {
		var statusCoder interface{ StatusCode() int }
		if errors.As(e.cause, &statusCoder) && statusCoder != nil {
			if code := statusCoder.StatusCode(); code > 0 {
				return code
			}
		}
		return 0
	}
	return e.status
}

func (e *claudeFastRequestError) IsRequestScoped() bool {
	if e == nil {
		return false
	}
	if e.IsCredentialScoped() {
		return false
	}
	return true
}

func (e *claudeFastRequestError) IsCredentialScoped() bool {
	if e == nil {
		return false
	}
	type credentialScopedProvider interface {
		IsCredentialScoped() bool
	}
	var csp credentialScopedProvider
	if errors.As(e.cause, &csp) && csp != nil {
		return csp.IsCredentialScoped()
	}
	return false
}

func (e *claudeFastRequestError) RetryAfter() *time.Duration {
	if e == nil {
		return nil
	}
	return e.retryAfter
}

// claudeFastDirectResponseError carries an upstream HTTP error response through
// the auth manager and protocol handlers without retrying or rebuilding its
// status and JSON body.
type claudeFastDirectResponseError struct {
	response         *cliproxyexecutor.RequestTerminatedError
	retryAfter       *time.Duration
	credentialScoped bool
	// modelScoped marks a rejection that belongs to the requested model on this
	// credential alone: neither request-scoped (which would record no state and stop
	// the credential loop) nor credential-scoped (which would bench a credential that
	// still serves every other model).
	modelScoped bool
}

func (e *claudeFastDirectResponseError) Error() string {
	if e == nil || e.response == nil {
		return ""
	}
	return fmt.Sprintf("claude Fast upstream request failed with status %d", e.response.HTTPStatus)
}

func (e *claudeFastDirectResponseError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.response
}

func (e *claudeFastDirectResponseError) IsRequestScoped() bool {
	if e == nil {
		return false
	}
	return !e.credentialScoped && !e.modelScoped
}

func (e *claudeFastDirectResponseError) IsCredentialScoped() bool {
	if e == nil {
		return false
	}
	return e.credentialScoped
}

func (e *claudeFastDirectResponseError) RetryAfter() *time.Duration {
	if e == nil {
		return nil
	}
	return e.retryAfter
}

func wrapClaudeFastRequestError(fastRequest bool, status int, err error) error {
	if err == nil || !fastRequest {
		return err
	}
	var retryAfter *time.Duration
	if rap, ok := err.(interface{ RetryAfter() *time.Duration }); ok && rap != nil {
		retryAfter = rap.RetryAfter()
	}
	return &claudeFastRequestError{cause: err, status: status, retryAfter: retryAfter}
}

func newClaudeFastDirectResponseError(resp *http.Response, body []byte) error {
	if resp == nil {
		return nil
	}
	headers := resp.Header.Clone()
	// body has already been decoded. Do not forward stale representation or
	// length headers that describe the compressed upstream bytes.
	headers.Del("Content-Encoding")
	headers.Del("Content-Length")

	var retryAfter *time.Duration
	credentialScoped, modelScoped := false, false
	if resp.StatusCode == http.StatusTooManyRequests {
		retryAfter = helps.ParseClaudeRateLimitResetForError(resp.Header, body, time.Now())
		if helps.ClaudeHeadersIndicateUnifiedRateLimitRejection(resp.Header) {
			credentialScoped = true
		} else if !claudeBodyIndicatesFastModeCredits(body) && helps.ClaudeHeadersIndicateFableOnlyRateLimitRejection(resp.Header) {
			// An exhausted Fable/overage allowance is a quota exhaustion on this one
			// account, not the fast-mode entitlement refusal the request-scoped policy
			// exists for: cool the requested model and let another pooled credential try.
			modelScoped = true
		}
	}

	return &claudeFastDirectResponseError{
		response: &cliproxyexecutor.RequestTerminatedError{
			HTTPStatus: resp.StatusCode,
			Header:     headers,
			Body:       bytes.Clone(body),
		},
		retryAfter:       retryAfter,
		credentialScoped: credentialScoped,
		modelScoped:      modelScoped,
	}
}

// logClaudeFableOnlyRejection records a Fable/overage-only 429 against the credential that
// returned it. The rejection cools one model on one account and nothing else, so without this
// line an exhausted Fable allowance leaves nothing in the log but a bare 429 access entry.
func logClaudeFableOnlyRejection(ctx context.Context, auth *cliproxyauth.Auth, model string, status int, headers http.Header) {
	if status != http.StatusTooManyRequests || !helps.ClaudeHeadersIndicateFableOnlyRateLimitRejection(headers) {
		return
	}
	buckets := make([]string, 0, 2)
	for _, window := range [...]string{"7d_oi", "Overage"} {
		if helps.ClaudeUnifiedWindowStatus(headers, window) == "rejected" {
			buckets = append(buckets, strings.ToLower(window))
		}
	}
	id, label, _, _ := claudeAuthLogIdentity(auth)
	fields := log.Fields{
		"auth_id":          id,
		"auth_label":       label,
		"model":            model,
		"rejected_buckets": strings.Join(buckets, ","),
		"retry_after":      headers.Get("Retry-After"),
		"reset":            claudeFableResetForLog(headers),
	}
	if auth != nil && auth.FileName != "" {
		fields["auth_file"] = filepath.Base(auth.FileName)
	}
	helps.LogWithRequestID(ctx).WithFields(fields).Warn("claude 429: Fable/overage allowance exhausted on this credential; cooling this model and rotating credentials")
}

func claudeFableResetForLog(headers http.Header) string {
	for _, name := range [...]string{"Anthropic-Ratelimit-Unified-7d_oi-Reset", "Anthropic-Ratelimit-Unified-Reset"} {
		if raw := strings.TrimSpace(headers.Get(name)); raw != "" {
			if epoch, errParse := strconv.ParseInt(raw, 10, 64); errParse == nil {
				return time.Unix(epoch, 0).UTC().Format(time.RFC3339)
			}
			return raw
		}
	}
	return ""
}

func claudeAuthLogIdentity(auth *cliproxyauth.Auth) (id, label, authType, authValue string) {
	if auth == nil {
		return "", "", "", ""
	}
	authType, authValue = auth.AccountInfo()
	return auth.ID, auth.Label, authType, authValue
}
