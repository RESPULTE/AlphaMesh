import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import type { AuthRequest, AuthResponse, ConversationBootstrapResponse } from '../types/api';

const STORAGE_ACCESS_TOKEN = 'alphamesh.auth.access_token';
const STORAGE_REFRESH_TOKEN = 'alphamesh.auth.refresh_token';
const STORAGE_USER_EMAIL = 'alphamesh.auth.user_email';
const STORAGE_SESSION_ID = 'alphamesh.active_session_id';
const STORAGE_CONVERSATION_ID = 'alphamesh.active_conversation_id';

type AuthMode = 'login' | 'signup';

interface AuthState {
  accessToken: string | null;
  refreshToken: string | null;
  userEmail: string | null;
  sessionId: string | null;
}

interface AuthContextValue {
  accessToken: string | null;
  refreshToken: string | null;
  userEmail: string | null;
  sessionId: string | null;
  isAuthenticated: boolean;
  authenticate: (mode: AuthMode, email: string) => Promise<void>;
  logout: () => Promise<void>;
  authFetch: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;
  buildAuthUrl: (path: string) => string;
}

function readInitialState(): AuthState {
  if (typeof window === 'undefined') {
    return {
      accessToken: null,
      refreshToken: null,
      userEmail: null,
      sessionId: null,
    };
  }
  // Force fresh login on every app load.
  window.localStorage.removeItem(STORAGE_ACCESS_TOKEN);
  window.localStorage.removeItem(STORAGE_REFRESH_TOKEN);
  window.localStorage.removeItem(STORAGE_USER_EMAIL);
  window.localStorage.removeItem(STORAGE_SESSION_ID);
  window.localStorage.removeItem(STORAGE_CONVERSATION_ID);
  return {
    accessToken: null,
    refreshToken: null,
    userEmail: null,
    sessionId: null,
  };
}

function writeAuthState(next: AuthState): void {
  if (typeof window === 'undefined') return;

  if (next.accessToken) {
    window.localStorage.setItem(STORAGE_ACCESS_TOKEN, next.accessToken);
  } else {
    window.localStorage.removeItem(STORAGE_ACCESS_TOKEN);
  }

  if (next.refreshToken) {
    window.localStorage.setItem(STORAGE_REFRESH_TOKEN, next.refreshToken);
  } else {
    window.localStorage.removeItem(STORAGE_REFRESH_TOKEN);
  }

  if (next.userEmail) {
    window.localStorage.setItem(STORAGE_USER_EMAIL, next.userEmail);
  } else {
    window.localStorage.removeItem(STORAGE_USER_EMAIL);
  }

  if (next.sessionId) {
    window.localStorage.setItem(STORAGE_SESSION_ID, next.sessionId);
  } else {
    window.localStorage.removeItem(STORAGE_SESSION_ID);
  }
}

function clearUserScopedState(): void {
  if (typeof window === 'undefined') return;
  window.localStorage.removeItem(STORAGE_CONVERSATION_ID);
  window.localStorage.removeItem(STORAGE_SESSION_ID);
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>(() => readInitialState());
  const refreshInFlightRef = useRef<Promise<string | null> | null>(null);

  const clearAuth = useCallback(() => {
    const cleared: AuthState = {
      accessToken: null,
      refreshToken: null,
      userEmail: null,
      sessionId: null,
    };
    setState(cleared);
    writeAuthState(cleared);
    clearUserScopedState();
  }, []);

  const applyAuthPayload = useCallback((payload: AuthResponse) => {
    const next: AuthState = {
      accessToken: payload.access_token,
      refreshToken: payload.refresh_token,
      userEmail: payload.user_email,
      sessionId: payload.session_id,
    };
    setState(next);
    writeAuthState(next);
  }, []);

  const authenticate = useCallback(
    async (mode: AuthMode, email: string) => {
      const body: AuthRequest = { email: email.trim().toLowerCase() };
      const response = await fetch(`/api/v1/auth/${mode}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        const detail = payload && typeof payload.detail === 'string' ? payload.detail : 'Authentication failed.';
        throw new Error(detail);
      }
      const payload = (await response.json()) as AuthResponse;
      try {
        const bootstrap = await fetch('/api/v1/conversations/bootstrap', {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${payload.access_token}`,
          },
        });
        // Backward compatibility: older backend builds may not expose this route yet.
        // Do not block login if bootstrap endpoint is unavailable.
        if (bootstrap.status === 404) {
          applyAuthPayload(payload);
          return;
        }
        if (!bootstrap.ok) {
          const bootstrapPayload = await bootstrap.json().catch(() => null);
          const detail =
            bootstrapPayload && typeof bootstrapPayload.detail === 'string'
              ? bootstrapPayload.detail
              : 'Unable to initialize conversation workspace.';
          throw new Error(`${detail} (HTTP ${bootstrap.status})`);
        }
        const bootstrapPayload = (await bootstrap.json()) as ConversationBootstrapResponse;
        if (bootstrapPayload.status !== 'ok') {
          throw new Error('Unable to initialize conversation workspace.');
        }
      } catch (err) {
        // Bootstrap failures should not block auth; first /chat call will still create chatlog.
        // Keep this warning so operational issues are visible during debugging.
        // eslint-disable-next-line no-console
        console.warn('Conversation workspace bootstrap failed; continuing login.', err);
      }
      applyAuthPayload(payload);
    },
    [applyAuthPayload]
  );

  const refreshAccessToken = useCallback(async (): Promise<string | null> => {
    if (!state.refreshToken) {
      clearAuth();
      return null;
    }

    if (refreshInFlightRef.current) {
      return refreshInFlightRef.current;
    }

    refreshInFlightRef.current = (async () => {
      try {
        const response = await fetch('/api/v1/auth/refresh', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ refresh_token: state.refreshToken }),
        });
        if (!response.ok) {
          clearAuth();
          return null;
        }
        const payload = (await response.json()) as AuthResponse;
        applyAuthPayload(payload);
        return payload.access_token;
      } catch {
        clearAuth();
        return null;
      } finally {
        refreshInFlightRef.current = null;
      }
    })();

    return refreshInFlightRef.current;
  }, [applyAuthPayload, clearAuth, state.refreshToken]);

  const withAuthHeaders = useCallback(
    (init: RequestInit = {}, token?: string | null): RequestInit => {
      const headers = new Headers(init.headers || {});
      if (token) {
        headers.set('Authorization', `Bearer ${token}`);
      }
      return { ...init, headers };
    },
    []
  );

  const authFetch = useCallback(
    async (input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> => {
      const first = await fetch(input, withAuthHeaders(init, state.accessToken));
      if (first.status !== 401) {
        return first;
      }

      const refreshedToken = await refreshAccessToken();
      if (!refreshedToken) {
        return first;
      }

      return fetch(input, withAuthHeaders(init, refreshedToken));
    },
    [refreshAccessToken, state.accessToken, withAuthHeaders]
  );

  const logout = useCallback(async () => {
    try {
      if (state.accessToken) {
        await fetch('/api/v1/auth/logout', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${state.accessToken}`,
          },
        });
      }
    } catch {
      // Best-effort only.
    } finally {
      clearAuth();
    }
  }, [clearAuth, state.accessToken]);

  const buildAuthUrl = useCallback(
    (path: string): string => {
      const token = state.accessToken;
      if (!token) return path;
      const separator = path.includes('?') ? '&' : '?';
      return `${path}${separator}token=${encodeURIComponent(token)}`;
    },
    [state.accessToken]
  );

  const value = useMemo<AuthContextValue>(
    () => ({
      accessToken: state.accessToken,
      refreshToken: state.refreshToken,
      userEmail: state.userEmail,
      sessionId: state.sessionId,
      isAuthenticated: Boolean(state.accessToken && state.refreshToken && state.userEmail),
      authenticate,
      logout,
      authFetch,
      buildAuthUrl,
    }),
    [authenticate, authFetch, buildAuthUrl, logout, state]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return ctx;
}
