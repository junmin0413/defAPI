<script setup>
import { ref, watch } from "vue";

const languages = [
  { value: "javascript", label: "JavaScript (.js, .jsx)" },
  { value: "typescript", label: "TypeScript (.ts, .tsx)" },
  { value: "python", label: "Python (.py)" },
  { value: "markup", label: "HTML/XML (.html)" },
  { value: "css", label: "CSS (.css)" },
  { value: "json", label: "JSON (.json)" },
  { value: "c", label: "C (.c)" },
  { value: "cpp", label: "C++ (.cpp)" },
  { value: "go", label: "Go (.go)" },
  { value: "bash", label: "Bash (.sh)" },
  { value: "yaml", label: "YAML (.yaml, .yml)" },
];

const props = defineProps({
  isOpen: {
    type: Boolean,
    required: true,
  },
});

const emit = defineEmits(["close", "save"]);

const codeContent = ref("");
const fileName = ref("");
const selectedLanguage = ref("javascript");
const isSubmitting = ref(false);
const error = ref(null);

watch(
  () => props.isOpen,
  (newVal) => {
    if (newVal) {
      codeContent.value = "";
      fileName.value = "";
      selectedLanguage.value = "javascript";
      error.value = null;
      isSubmitting.value = false;
    }
  }
);

const closeModal = () => {
  emit("close");
};

const handleSave = () => {
  if (!fileName.value.trim()) {
    error.value = "파일 이름을 입력해주세요.";
    return;
  }
  if (!codeContent.value.trim()) {
    error.value = "코드를 입력해주세요.";
    return;
  }

  isSubmitting.value = true;
  error.value = null;

  try {
    const fileData = {
      name: fileName.value.trim(),
      language: selectedLanguage.value,
      content: codeContent.value,
    };

    // Simulate async operation for a moment
    setTimeout(() => {
      emit("save", fileData);
      // Resetting state and closing will happen in the parent component
      // if they handle the save event by setting isOpen to false.
      isSubmitting.value = false;
    }, 100);
  } catch (e) {
    error.value = `저장 중 오류가 발생했습니다: ${e.message}`;
    isSubmitting.value = false;
  }
};
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
        <h3 class="text-lg font-medium text-stone-300 truncate pl-1">
          새 코드 입력 (수동 작성)
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

      <!-- Modal Body (Form) -->
      <div class="p-4 overflow-y-auto grow scrollbar-thin">
        <form @submit.prevent="handleSave" class="space-y-4">
          <!-- File Details -->
          <div class="flex flex-col sm:flex-row gap-4">
            <div class="flex-1">
              <label
                for="fileName"
                class="block text-sm font-medium text-stone-400 mb-1"
                >파일 이름 (예: `script.js`)</label
              >
              <input
                type="text"
                id="fileName"
                v-model="fileName"
                required
                placeholder="filename.extension"
                :class="[
                  'w-full p-2.5 bg-stone-800 text-stone-200 border rounded-lg focus:ring-pink-500 focus:border-pink-500',
                  error && !fileName.trim()
                    ? 'border-pink-500'
                    : 'border-stone-700',
                ]"
              />
            </div>

            <div class="sm:w-1/3">
              <label
                for="languageSelect"
                class="block text-sm font-medium text-stone-400 mb-1"
                >언어/하이라이트</label
              >
              <select
                id="languageSelect"
                v-model="selectedLanguage"
                class="w-full p-2.5 bg-stone-800 text-stone-200 border border-stone-700 rounded-lg focus:ring-pink-500 focus:border-pink-500 appearance-none cursor-pointer"
              >
                <option
                  v-for="lang in languages"
                  :key="lang.value"
                  :value="lang.value"
                >
                  {{ lang.label }}
                </option>
              </select>
            </div>
          </div>

          <!-- Code Input Area -->
          <div>
            <label
              for="codeInput"
              class="block text-sm font-medium text-stone-400 mb-1"
              >코드 내용</label
            >
            <textarea
              id="codeInput"
              v-model="codeContent"
              required
              rows="15"
              placeholder="여기에 코드를 입력하세요..."
              class="w-full p-4 text-sm font-mono bg-stone-800 text-stone-200 border border-stone-700 rounded-lg focus:ring-pink-500 focus:border-pink-500 resize-none scrollbar-thin"
              :class="{
                'border-pink-500': error && !codeContent.trim(),
              }"
            ></textarea>
          </div>

          <!-- Error Message -->
          <div
            v-if="error"
            class="text-pink-400 bg-pink-900/20 border border-pink-500/30 px-4 py-2 rounded-lg text-sm"
          >
            {{ error }}
          </div>
        </form>
      </div>

      <!-- Modal Footer (Save Button) -->
      <div
        class="flex justify-end p-4 border-t border-stone-700 bg-stone-900 rounded-b-xl"
      >
        <button
          @click="handleSave"
          :disabled="isSubmitting"
          class="px-6 py-2 text-white font-semibold rounded-lg transition duration-200 shadow-md flex items-center justify-center"
          :class="{
            'bg-pink-600 hover:bg-pink-700 focus:ring-4 focus:ring-pink-500 focus:ring-opacity-50':
              !isSubmitting,
            'bg-pink-800 cursor-not-allowed': isSubmitting,
          }"
        >
          <svg
            v-if="isSubmitting"
            class="animate-spin -ml-1 mr-3 size-5 text-white"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              class="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              stroke-width="4"
            ></circle>
            <path
              class="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            ></path>
          </svg>
          {{ isSubmitting ? "저장 중..." : "저장 및 분석 시작" }}
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* Scrollbar styles reused from FileViewerModal */
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
</style>
