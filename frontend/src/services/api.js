import { computed, reactive } from "vue";

const STORAGE_KEY = "defapi-auth";
const defaultBase = import.meta.env.VITE_API_BASE_URL || "/api";
const fallbackBase =
  import.meta.env.VITE_API_BASE_URL_FALLBACK || "http://localhost:8000/api";

const normalizeBase = (value) =>
  value.endsWith("/") ? value.slice(0, -1) : value;

const API_BASE_URL = normalizeBase(defaultBase);
const API_BASE_FALLBACK = normalizeBase(fallbackBase);

const authState = reactive({
  user: null,
  accessToken: null,
  refreshToken: null,
});

const isSignedIn = computed(
  () => Boolean(authState.user && authState.accessToken)
);

const persistAuth = () => {
  const payload = {
    user: authState.user,
    tokens: {
      accessToken: authState.accessToken,
      refreshToken: authState.refreshToken,
    },
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
};

const restoreAuth = () => {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    authState.user = parsed.user || null;
    authState.accessToken = parsed.tokens?.accessToken || null;
    authState.refreshToken = parsed.tokens?.refreshToken || null;
  } catch (error) {
    console.warn("인증 정보 로드 실패", error);
    clearAuth();
  }
};

const clearAuth = () => {
  authState.user = null;
  authState.accessToken = null;
  authState.refreshToken = null;
  localStorage.removeItem(STORAGE_KEY);
};

const setAuth = ({ user, tokens }) => {
  authState.user = user || null;
  authState.accessToken = tokens?.accessToken || null;
  authState.refreshToken = tokens?.refreshToken || null;
  persistAuth();
};

const parseError = async (response) => {
  try {
    const data = await response.json();
    return data?.message || response.statusText;
  } catch (_err) {
    return response.statusText || "요청에 실패했습니다.";
  }
};

const refreshTokens = async () => {
  if (!authState.refreshToken) return null;

  const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refreshToken: authState.refreshToken }),
  });

  if (!response.ok) {
    clearAuth();
    return null;
  }

  const data = await response.json();
  if (!data?.tokens?.accessToken) {
    clearAuth();
    return null;
  }

  authState.accessToken = data.tokens.accessToken;
  authState.refreshToken =
    data.tokens.refreshToken || authState.refreshToken || null;
  persistAuth();
  return data.tokens;
};

const performFetch = async (baseUrl, path, { method, headers, body }) => {
  const url = `${baseUrl}${path}`;
  return fetch(url, {
    method,
    headers,
    body:
      body instanceof FormData
        ? body
        : body
          ? JSON.stringify(body)
          : undefined,
  });
};

const apiRequest = async (
  path,
  { method = "GET", body, headers = {}, auth = true, retry = true } = {}
) => {
  const finalHeaders = new Headers(headers);

  if (!(body instanceof FormData)) {
    if (!finalHeaders.has("Content-Type")) {
      finalHeaders.set("Content-Type", "application/json");
    }
  }

  if (auth && authState.accessToken) {
    finalHeaders.set("Authorization", `Bearer ${authState.accessToken}`);
  }

  const tryBases = [API_BASE_URL];
  if (API_BASE_FALLBACK && API_BASE_FALLBACK !== API_BASE_URL) {
    tryBases.push(API_BASE_FALLBACK);
  }

  let lastError = null;
  for (const base of tryBases) {
    let response;
    try {
      response = await performFetch(base, path, {
        method,
        headers: finalHeaders,
        body,
      });
    } catch (error) {
      lastError = error;
      continue;
    }

    if (response.status === 401 && auth && retry && authState.refreshToken) {
      const refreshed = await refreshTokens();
      if (refreshed) {
        return apiRequest(path, { method, body, headers, auth, retry: false });
      }
    }

    if (!response.ok) {
      const message = await parseError(response);
      lastError = new Error(message || "요청 실패");
      continue;
    }

    if (response.status === 204) return null;
    try {
      return await response.json();
    } catch (_err) {
      return null;
    }
  }

  throw lastError || new Error("네트워크 요청 중 문제가 발생했습니다.");
};

export const login = async (email, password) => {
  const data = await apiRequest(
    "/auth/login",
    {
      method: "POST",
      body: { email, password },
      auth: false,
    }
  );
  if (data?.user && data?.tokens) {
    setAuth(data);
  }
  return data;
};

export const signup = async ({ email, password, displayName }) => {
  const data = await apiRequest("/auth/signup", {
    method: "POST",
    body: { email, password, displayName },
    auth: false,
  });
  if (data?.user && data?.tokens) {
    setAuth(data);
  }
  return data;
};

export const logout = async () => {
  try {
    if (authState.refreshToken) {
      await apiRequest("/auth/logout", {
        method: "POST",
        body: { refreshToken: authState.refreshToken },
      });
    }
  } catch (error) {
    console.warn("로그아웃 API 실패", error);
  } finally {
    clearAuth();
  }
};

export const loginWithGoogle = async (idToken) => {
  const data = await apiRequest("/auth/oauth/google", {
    method: "POST",
    body: { idToken },
    auth: false,
  });
  if (data?.user && data?.tokens) {
    setAuth(data);
  }
  return data;
};

export const loginWithGithub = async (code) => {
  const data = await apiRequest("/auth/oauth/github", {
    method: "POST",
    body: { code },
    auth: false,
  });
  if (data?.user && data?.tokens) {
    setAuth(data);
  }
  return data;
};

export const getProjects = async () => {
  const data = await apiRequest("/projects");
  return data?.projects || [];
};

export const createProject = async ({
  name,
  language = "unknown",
  description = "",
  visibility = "private",
}) => {
  const data = await apiRequest("/projects", {
    method: "POST",
    body: { name, language, description, visibility },
  });
  return data?.project;
};

export const uploadArtifact = async (projectId, file) => {
  const formData = new FormData();
  formData.append("artifact", file);

  return apiRequest(`/projects/${projectId}/upload`, {
    method: "POST",
    body: formData,
    headers: {},
  });
};

export const startScan = async (projectId, { inputType, content, filePath }) => {
  return apiRequest(`/projects/${projectId}/scan`, {
    method: "POST",
    body: { inputType, content, filePath },
  });
};

export const getScanStatus = async (scanId) => {
  const data = await apiRequest(`/scans/${scanId}`);
  return data?.scan;
};

export const getReport = async (reportId) => {
  return apiRequest(`/reports/${reportId}`);
};

export const getOrCreateDefaultProject = async () => {
  const projects = await getProjects();
  if (projects.length > 0) return projects[0];

  const timestamp = new Date();
  const name = `project-${timestamp.getFullYear()}${String(
    timestamp.getMonth() + 1
  ).padStart(2, "0")}${String(timestamp.getDate()).padStart(2, "0")}`;

  return createProject({ name, language: "auto", description: "자동 생성됨" });
};

export { authState, clearAuth, isSignedIn, restoreAuth, setAuth };

restoreAuth();
