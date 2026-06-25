<script setup>
import { onMounted, ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import { loginWithGoogle, loginWithGithub } from "@/services/api";

const route = useRoute();
const router = useRouter();

const status = ref("처리 중...");
const errorMessage = ref("");

const getState = () => localStorage.getItem("oauth_state");
const clearState = () => localStorage.removeItem("oauth_state");

const parseFragment = () => {
  const hash = window.location.hash || "";
  const params = new URLSearchParams(hash.replace(/^#/, ""));
  const result = {};
  params.forEach((v, k) => {
    result[k] = v;
  });
  return result;
};

const handleGoogle = async (idToken, state) => {
  const saved = getState();
  if (saved && state && saved !== state) {
    throw new Error("Google OAuth state 검증에 실패했습니다.");
  }
  clearState();
  await loginWithGoogle(idToken);
};

const handleGithub = async (code, state) => {
  const saved = getState();
  if (saved && state && saved !== state) {
    throw new Error("GitHub OAuth state 검증에 실패했습니다.");
  }
  clearState();
  await loginWithGithub(code);
};

onMounted(async () => {
  try {
    const fragment = parseFragment();
    const query = route.query;

    // Google implicit flow: id_token in fragment
    if (fragment.id_token) {
      status.value = "Google 로그인 처리 중...";
      await handleGoogle(fragment.id_token, fragment.state || query.state);
    }
    // GitHub code flow: code in query
    else if (query.code) {
      status.value = "GitHub 로그인 처리 중...";
      await handleGithub(query.code, query.state);
    } else {
      throw new Error("OAuth 코드나 토큰을 찾을 수 없습니다.");
    }

    const redirectTo = route.query.redirect || "/analysis";
    router.replace(redirectTo);
  } catch (err) {
    errorMessage.value = err?.message || "소셜 로그인 처리 중 오류가 발생했습니다.";
    status.value = "로그인 실패";
  }
});
</script>

<template>
  <div
    class="min-h-screen flex flex-col items-center justify-center gap-4 bg-stone-900 text-stone-200"
  >
    <span class="text-lg font-semibold">{{ status }}</span>
    <p v-if="errorMessage" class="text-pink-400 text-sm text-center px-4">
      {{ errorMessage }}
    </p>
  </div>
</template>
