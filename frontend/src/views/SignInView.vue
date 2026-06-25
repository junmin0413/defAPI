<script setup>
import { ref } from "vue";
import { RouterLink, useRoute, useRouter } from "vue-router";
import { login, loginWithGithub, loginWithGoogle } from "@/services/api";

const email = ref("");
const password = ref("");
const errorMessage = ref("");
const socialError = ref("");
const loading = ref(false);
const googleLoading = ref(false);
const githubLoading = ref(false);

const router = useRouter();
const route = useRoute();

const googleRedirectUri =
  import.meta.env.VITE_GOOGLE_REDIRECT_URL ||
  import.meta.env.VITE_OAUTH_REDIRECT_URL ||
  `${window.location.origin}/oauth/callback`;
const githubRedirectUri =
  import.meta.env.VITE_GITHUB_REDIRECT_URL ||
  import.meta.env.VITE_OAUTH_REDIRECT_URL ||
  `${window.location.origin}/oauth/callback`;
const googleClientId =
  import.meta.env.VITE_GOOGLE_CLIENT_ID ||
  "112914357105-fm7jklu3us7pi734cfcs7o4bupmbinmh.apps.googleusercontent.com";
const githubClientId =
  import.meta.env.VITE_GITHUB_CLIENT_ID || "Ov23liOS5Xq8VhrnC1fk";

const generateState = () =>
  (window.crypto?.randomUUID?.() ||
    Math.random().toString(36).slice(2) + Date.now().toString(36));

const handleSubmit = async () => {
  if (loading.value) return;
  errorMessage.value = "";
  loading.value = true;
  try {
    await login(email.value.trim(), password.value);
    const redirectTo = route.query.redirect || "/analysis";
    router.push(redirectTo);
  } catch (error) {
    errorMessage.value = error?.message || "로그인에 실패했습니다.";
  } finally {
    loading.value = false;
  }
};

const handleGoogleLogin = () => {
  if (googleLoading.value) return;
  socialError.value = "";
  const state = generateState();
  localStorage.setItem("oauth_state", state);
  const url = new URL("https://accounts.google.com/o/oauth2/v2/auth");
  url.searchParams.set("client_id", googleClientId);
  url.searchParams.set("redirect_uri", googleRedirectUri);
  url.searchParams.set("response_type", "id_token");
  url.searchParams.set("scope", "openid email profile");
  url.searchParams.set("nonce", state);
  url.searchParams.set("state", state);
  url.searchParams.set("prompt", "select_account");
  window.location.href = url.toString();
};

const handleGithubLogin = () => {
  if (githubLoading.value) return;
  socialError.value = "";
  const state = generateState();
  localStorage.setItem("oauth_state", state);
  const url = new URL("https://github.com/login/oauth/authorize");
  url.searchParams.set("client_id", githubClientId);
  url.searchParams.set("redirect_uri", githubRedirectUri);
  url.searchParams.set("scope", "read:user user:email");
  url.searchParams.set("state", state);
  window.location.href = url.toString();
};
</script>

<template>
  <div
    class="flex flex-col justify-evenly items-center gap-6 py-16 md:flex-row md:py-0"
  >
    <RouterLink
      class="group absolute top-8 left-8 flex justify-center items-center size-10 cursor-pointer"
      title="돌아가기"
      to="/"
    >
      <svg
        class="size-8 fill-stone-500 group-hover:fill-stone-400 transition ease-in-out duration-200"
        viewBox="0 0 24 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <g id="SVGRepo_bgCarrier" stroke-width="0"></g>
        <g
          id="SVGRepo_tracerCarrier"
          stroke-linecap="round"
          stroke-linejoin="round"
        ></g>
        <g id="SVGRepo_iconCarrier">
          <path
            d="M21.2916 17.0218L19.5516 13.5418C19.0716 12.5718 19.0716 11.4418 19.5516 10.4718L21.2916 6.99178C22.7816 4.00178 19.5716 0.861782 16.6216 2.42178L15.0816 3.24178C14.8716 3.35178 14.7016 3.53178 14.6016 3.75178L8.90164 16.3918C8.67164 16.9118 8.88164 17.5218 9.38164 17.7818L16.6216 21.5918C19.5716 23.1518 22.7816 20.0018 21.2916 17.0218Z"
          ></path>
          <path
            d="M7.68953 15.6004L11.4195 7.32045C11.8395 6.39045 10.8395 5.45045 9.93953 5.93045L4.16953 8.96045C1.71953 10.2504 1.71953 13.7504 4.16953 15.0404L6.20953 16.1104C6.75953 16.4004 7.42953 16.1704 7.68953 15.6004Z"
          ></path>
        </g>
      </svg>
    </RouterLink>
    <span class="text-6xl cursor-default">SIGN IN</span>
    <section
      class="flex flex-col justify-evenly items-center w-full max-w-xl h-dvh md:py-8"
    >
      <form
        method="post"
        @submit.prevent="handleSubmit"
        class="flex flex-col justify-evenly items-center gap-16 w-full px-4 md:px-16"
      >
        <div class="flex flex-col gap-3 w-full">
          <label class="text-2xl" for="user-email">EMAIL</label>
          <input
            class="w-full h-16 border-2 border-stone-500 rounded-full px-7 text-stone-400 placeholder:text-stone-500 focus:outline-none focus:ring-2 focus:ring-pink-500 caret-pink-500"
            id="user-email"
            type="email"
            placeholder="계정 이메일을 입력하세요."
            v-model.trim="email"
            required
          />
        </div>
        <div class="flex flex-col gap-3 w-full">
          <label class="text-2xl" for="user-email">PASSWORD</label>
          <input
            class="w-full h-16 border-2 border-stone-500 rounded-full px-7 text-stone-400 placeholder:text-stone-500 focus:outline-none focus:ring-2 focus:ring-pink-500 caret-pink-500"
            id="user-password"
            type="password"
            placeholder="계정 비밀번호를 입력하세요."
            v-model="password"
            required
          />
        </div>
        <button
          class="w-full h-16 mine-w-80 transition ease-in-out duration-200 text-lg border-2 border-transparent rounded-full cursor-pointer hover:text-stone-400"
          type="submit"
          :disabled="loading"
        >
          {{ loading ? "로그인 중..." : "로그인" }}
        </button>
        <p v-if="errorMessage" class="text-sm text-pink-400 text-center">
          {{ errorMessage }}
        </p>
      </form>
      <div class="flex flex-col items-center gap-16 w-full">
        <p class="text-base cursor-default">
          계정이 없다면,
          <RouterLink
            class="underline text-stone-400 cursor-pointer"
            to="/sign-up"
            >여기서 회원가입</RouterLink
          >
        </p>
        <div class="w-10/12 outline outline-stone-500"></div>
        <div class="flex flex-col gap-3 items-center">
          <p v-if="socialError" class="text-sm text-pink-400 text-center">
            {{ socialError }}
          </p>
        </div>
        <div class="flex justify-center gap-7" style="margin-top: -40px;">
          <button
            @click.prevent="handleGoogleLogin"
            :disabled="googleLoading"
            class="group flex justify-center items-center size-16 rounded-full border-2 cursor-pointer border-stone-500 hover:border-stone-400 transition ease-in-out duration-200"
            title="Google 계정으로 로그인"
          >
            <svg
              class="size-8 fill-stone-500 group-hover:fill-stone-400 transition ease-in-out duration-200"
              viewBox="0 0 20 20"
              xmlns="http://www.w3.org/2000/svg"
            >
              <g id="SVGRepo_bgCarrier" stroke-width="0"></g>
              <g
                id="SVGRepo_tracerCarrier"
                stroke-linecap="round"
                stroke-linejoin="round"
              ></g>
              <g id="SVGRepo_iconCarrier">
                <rect x="0" fill="none" width="20" height="20"></rect>
                <g>
                  <path
                    d="M17.6 8.5h-7.5v3h4.4c-.4 2.1-2.3 3.5-4.4 3.4-2.6-.1-4.6-2.1-4.7-4.7-.1-2.7 2-5 4.7-5.1 1.1 0 2.2.4 3.1 1.2l2.3-2.2C14.1 2.7 12.1 2 10.2 2c-4.4 0-8 3.6-8 8s3.6 8 8 8c4.6 0 7.7-3.2 7.7-7.8-.1-.6-.1-1.1-.3-1.7z"
                  ></path>
                </g>
              </g>
            </svg>
          </button>
          <button
            @click.prevent="handleGithubLogin"
            :disabled="githubLoading"
            class="group flex justify-center items-center size-16 rounded-full border-2 cursor-pointer border-stone-500 hover:border-stone-400 transition ease-in-out duration-200"
            title="GitHub 계정으로 로그인"
          >
            <svg
              class="size-8 fill-stone-500 group-hover:fill-stone-400 transition ease-in-out duration-200"
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <g id="SVGRepo_bgCarrier" stroke-width="0"></g>
              <g
                id="SVGRepo_tracerCarrier"
                stroke-linecap="round"
                stroke-linejoin="round"
              ></g>
              <g id="SVGRepo_iconCarrier">
                <path
                  d="M18.6713 2.62664C18.5628 2.36483 18.3534 2.16452 18.0959 2.07627L18.094 2.07564L18.0922 2.07501L18.0884 2.07374L18.0805 2.07114L18.0636 2.06583C18.0518 2.06223 18.039 2.05856 18.0252 2.05487C17.9976 2.04749 17.966 2.04007 17.9305 2.03319C17.8593 2.01941 17.7728 2.00787 17.6708 2.00279C17.466 1.99259 17.2037 2.00858 16.8817 2.08054C16.3447 2.20053 15.6476 2.47464 14.7724 3.03631C14.7152 3.07302 14.6572 3.11096 14.5985 3.15016C14.5397 3.13561 14.4809 3.12155 14.422 3.108C12.8261 2.74083 11.1742 2.74083 9.57825 3.108C9.51933 3.12156 9.46049 3.13561 9.40173 3.15017C9.34298 3.11096 9.28499 3.07302 9.22775 3.03631C8.35163 2.47435 7.65291 2.20029 7.11455 2.08039C6.79179 2.00852 6.52891 1.99262 6.324 2.00278C6.22186 2.00784 6.13536 2.01931 6.06428 2.03299C6.0288 2.03982 5.99732 2.04717 5.96983 2.05447C5.95609 2.05812 5.94336 2.06176 5.93163 2.06531L5.91481 2.07056L5.90698 2.07311L5.9032 2.07437L5.90135 2.07499L5.89952 2.07561C5.63979 2.16397 5.42877 2.36623 5.32049 2.63061C4.91716 3.6154 4.8101 4.70134 5.00435 5.74306C5.01379 5.79367 5.02394 5.84418 5.0348 5.89458C4.99316 5.95373 4.9527 6.01368 4.91343 6.07439C4.30771 7.01089 3.98553 8.12791 4.00063 9.27493C4.00208 11.7315 4.71965 13.4139 5.9332 14.4965C6.62014 15.1093 7.41743 15.4844 8.21873 15.7208C8.31042 15.7479 8.40217 15.7731 8.49381 15.7967C8.48043 15.8432 8.46796 15.8901 8.45641 15.9373C8.40789 16.1357 8.37572 16.3394 8.36083 16.5461C8.35948 16.5648 8.35863 16.5835 8.35829 16.6022L8.32436 18.421L8.32417 18.4407C8.32417 18.4464 8.32417 18.4521 8.32417 18.4577C8.26262 18.473 8.20005 18.4843 8.13682 18.4916C7.942 18.5141 7.74467 18.4977 7.5561 18.4434C7.36752 18.3891 7.19127 18.2979 7.03752 18.1749C6.88377 18.0519 6.75553 17.8994 6.66031 17.7261L6.6505 17.7087C6.38836 17.2535 6.02627 16.8639 5.59142 16.5695C5.15656 16.275 4.6604 16.0836 4.14047 16.0099C3.59365 15.9324 3.08753 16.3128 3.01002 16.8597C2.93251 17.4065 3.31296 17.9126 3.85978 17.9901C4.07816 18.0211 4.28688 18.1015 4.47012 18.2256C4.65121 18.3482 4.80277 18.5103 4.9134 18.7C5.1346 19.0992 5.43165 19.4514 5.78801 19.7365C6.14753 20.0242 6.56032 20.2379 7.00272 20.3653C7.43348 20.4893 7.88392 20.5291 8.32949 20.4825C8.33039 20.7224 8.33103 20.9065 8.33103 21C8.33103 21.5523 8.75521 22 9.27847 22H14.7558C15.279 22 15.7032 21.5523 15.7032 21V17.2095C15.729 16.7802 15.685 16.3499 15.5738 15.9373C15.5585 15.8805 15.5419 15.824 15.5241 15.7679C15.5838 15.753 15.6435 15.7373 15.7032 15.7208C16.5277 15.4937 17.3513 15.1224 18.0588 14.4983C19.2791 13.4217 19.9982 11.7379 19.9996 9.27493C20.0147 8.12791 19.6925 7.01089 19.0868 6.07439C19.0475 6.01358 19.007 5.95354 18.9652 5.89429C18.976 5.84399 18.9861 5.79358 18.9955 5.74306C19.1893 4.69934 19.0795 3.61142 18.6713 2.62664Z"
                ></path>
              </g>
            </svg>
          </button>
        </div>
      </div>
    </section>
  </div>
</template>
