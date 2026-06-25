<script setup>
import { RouterView } from "vue-router";
import { provide, onMounted } from "vue";
import { authState, isSignedIn, restoreAuth } from "@/services/api";

provide("auth-state", authState);
provide("is-signed-in", isSignedIn);

onMounted(() => {
  restoreAuth();
});
</script>

<template>
  <RouterView />
</template>

