<script setup>
import { ref, watch, nextTick, onMounted } from "vue";
import Prism from "prismjs";

import "prismjs/themes/prism-tomorrow.css";
import "prismjs/components/prism-javascript";
import "prismjs/components/prism-typescript"; // includes tsx
import "prismjs/components/prism-python";
import "prismjs/components/prism-markup"; // html, xml, svg
import "prismjs/components/prism-css";
import "prismjs/components/prism-json";
import "prismjs/components/prism-c";
import "prismjs/components/prism-cpp";
import "prismjs/components/prism-go";
import "prismjs/components/prism-bash";
import "prismjs/components/prism-yaml";
import "prismjs/components/prism-kotlin";

const props = defineProps({
  file: {
    type: Object,
    required: true,
  },
  isOpen: {
    type: Boolean,
    required: true,
  },
});

const emit = defineEmits(["close"]);

const fileContent = ref("파일 내용 읽는 중...");
const fileReadError = ref(null);
const isLoading = ref(false);
const codeRef = ref(null);

const highlightCode = () => {
  if (codeRef.value) {
    Prism.highlightElement(codeRef.value);
  }
};

const readFileContent = () => {
  if (!props.file || !props.file.file) {
    fileReadError.value = "파일 데이터가 유효하지 않습니다.";
    return;
  }

  isLoading.value = true;
  const reader = new FileReader();

  reader.onload = (e) => {
    fileContent.value = e.target.result;
    fileReadError.value = null;
    isLoading.value = false;

    nextTick(() => {
      highlightCode();
    });
  };

  reader.onerror = () => {
    fileReadError.value = `파일을 읽는 중 오류가 발생했습니다: ${reader.error.name}`;
    fileContent.value = "";
    isLoading.value = false;
    console.error(fileReadError.value, reader.error);
  };

  try {
    reader.readAsText(props.file.file);
  } catch (e) {
    fileReadError.value = `파일 읽기 오류: ${e.message}`;
    fileContent.value = "";
    isLoading.value = false;
  }
};

watch(
  () => props.isOpen,
  (newVal) => {
    if (newVal) {
      fileContent.value = "파일 내용 읽는 중...";
      fileReadError.value = null;
      readFileContent();
    }
  }
);

const closeModal = () => {
  emit("close");
};

const getLanguage = (fileName) => {
  if (!fileName) return "plaintext";
  const parts = fileName.split(".");
  if (parts.length > 1) {
    const ext = parts.pop().toLowerCase();
    const map = {
      js: "javascript",
      ts: "typescript",
      jsx: "javascript", // Prism parses JSX inside js usually, or requires jsx component
      tsx: "typescript",
      py: "python",
      java: "java",
      c: "c",
      cc: "cpp",
      cpp: "cpp",
      cxx: "cpp",
      h: "cpp",
      hh: "cpp",
      hpp: "cpp",
      hxx: "cpp",
      kt: "kotlin",
      kts: "kotlin",
      cs: "csharp",
      rb: "ruby",
      go: "go",
      php: "php",
      html: "markup", // Prism uses 'markup' for HTML
      css: "css",
      json: "json",
      xml: "markup",
      yaml: "yaml",
      yml: "yaml",
      vue: "markup", // Vue SFCs are treated mostly as HTML markup
      sh: "bash",
      md: "markdown",
    };
    return map[ext] || "plaintext";
  }
  return "plaintext";
};

onMounted(() => {
  if (props.isOpen && fileContent.value !== "파일 내용 읽는 중...") {
    nextTick(highlightCode);
  }
});
</script>

<template>
  <div
    v-if="props.isOpen"
    class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-stone-900/10 backdrop-blur-sm"
    @click.self="closeModal"
  >
    <div
      class="relative w-full max-w-4xl max-h-[90vh] bg-stone-900 border border-stone-700 rounded-xl flex flex-col shadow-2xl"
    >
      <!-- Modal Header -->
      <div
        class="flex items-center justify-between p-4 border-b border-stone-700 bg-stone-900 rounded-t-xl"
      >
        <h3 class="text-lg font-medium text-stone-300 truncate pr-8 pl-1">
          {{ props.file.name }}
        </h3>
        <button
          @click="closeModal"
          class="p-2 rounded-full text-stone-400 hover:bg-stone-700 hover:text-white cursor-pointer"
          aria-label="Close modal"
        >
          <svg
            class="size-5"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>
      </div>

      <!-- Modal Body -->
      <div class="p-0 overflow-y-auto grow scrollbar-thin rounded-b-xl">
        <div
          v-if="isLoading"
          class="flex flex-col items-center justify-center p-12 text-stone-400 h-64"
        >
          <div
            class="animate-spin inline-block size-8 border-2 border-stone-500 border-t-pink-500 rounded-full mb-4"
            role="status"
          >
            <span class="sr-only">Loading...</span>
          </div>
          <p class="text-sm">파일 내용 읽는 중...</p>
        </div>

        <div v-else-if="fileReadError" class="p-8 flex justify-center">
          <div
            class="text-pink-400 bg-pink-900/20 border border-pink-500/30 px-4 py-3 rounded-lg text-sm"
          >
            {{ fileReadError }}
          </div>
        </div>

        <pre
          v-else
          class="m-0! p-4! min-h-full text-sm font-mono"
        ><code ref="codeRef" :class="`language-${getLanguage(props.file.name)}`">{{ fileContent }}</code></pre>
      </div>
    </div>
  </div>
</template>

<style scoped>
.scrollbar-thin::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}
.scrollbar-thin::-webkit-scrollbar-track {
  background: oklch(26.8% 0.007 34.298); /* stone-800 */
}
.scrollbar-thin::-webkit-scrollbar-thumb {
  background: oklch(55.3% 0.013 58.071); /* stone-600 */
  border-radius: 4px;
}
.scrollbar-thin::-webkit-scrollbar-thumb:hover {
  background-color: oklch(55.3% 0.013 58.071); /* stone-500 */
}

pre[class*="language-"] {
  background-color: oklch(21.6% 0.006 56.043) !important; /* stone-900 */
  text-shadow: none !important;
  font-family: Consolas, Monaco, "Andale Mono", "Ubuntu Mono", monospace;
}
code[class*="language-"] {
  text-shadow: none !important;
}
</style>
