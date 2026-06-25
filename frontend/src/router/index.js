import { createRouter, createWebHistory } from "vue-router";
import HomeView from "@/views/HomeView.vue";
import SignInView from "@/views/SignInView.vue";
import SignUpView from "@/views/SignUpView.vue";
import AnalysisView from "@/views/AnalysisView.vue";
import OAuthCallbackView from "@/views/OAuthCallbackView.vue";
import NotFoundView from "@/views/NotFoundView.vue";
import { isSignedIn } from "@/services/api";

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "/",
      name: "home",
      component: HomeView,
    },
    {
      path: "/analysis",
      name: "security-analysis",
      component: AnalysisView,
      meta: { requiresAuth: true },
    },
    {
      path: "/oauth/callback",
      name: "oauth-callback",
      component: OAuthCallbackView,
    },
    {
      path: "/auth/google/callback",
      name: "oauth-google-callback",
      component: OAuthCallbackView,
    },
    {
      path: "/auth/github/callback",
      name: "oauth-github-callback",
      component: OAuthCallbackView,
    },
    {
      path: "/sign-in",
      name: "sign-in",
      component: SignInView,
    },
    {
      path: "/sign-up",
      name: "sign-up",
      component: SignUpView,
    },
    {
      path: "/:catchAll(.*)",
      name: "not-found",
      component: NotFoundView,
    },
  ],
});

router.beforeEach((to, _from, next) => {
  if (to.meta.requiresAuth && !isSignedIn.value) {
    next({ name: "sign-in", query: { redirect: to.fullPath } });
    return;
  }
  next();
});

export default router;
