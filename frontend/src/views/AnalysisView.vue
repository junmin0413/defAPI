<script setup>
import { onBeforeRouteLeave } from "vue-router";
import { ref, onMounted, onBeforeUnmount, computed } from "vue";
import Navbar from "@/components/Navbar.vue";
import UploadedFile from "@/components/UploadedFile.vue";
import UploadedFileModal from "@/components/UploadedFileModal.vue";
import {
  getOrCreateDefaultProject,
  startScan,
  getScanStatus,
  getReport,
} from "@/services/api";

// PrismJS
import Prism from "prismjs";
import "prismjs/themes/prism-tomorrow.css";
// 언어
import "prismjs/components/prism-javascript";
import "prismjs/components/prism-typescript"; // tsx 포함
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

// 상태
const isFirstInput = ref(true);
const awaitingResponse = ref(false);
const isCopied = ref(false);
const inputType = ref("upload");
const errorMessage = ref("");
const currentProject = ref(null);

const loadingTexts = [
  "보안 취약점 분석 중.",
  "보안 취약점 분석 중..",
  "보안 취약점 분석 중...",
];
const loadingMsgIndex = ref(0);
const loadingText = ref(loadingTexts[0]);
let loadingInterval = null;

const analysisResult = ref(null);
const currentIssueIndex = ref(0);

const file = ref(null);
const message = ref("");
const messageTitle = ref("");
const fileInputRef = ref(null);
const submittedSource = ref(null);
const isSourceModalOpen = ref(false);

// 로딩 애니메이션
const startLoadingAnimation = () => {
  if (loadingInterval) clearInterval(loadingInterval);
  loadingInterval = setInterval(() => {
    loadingMsgIndex.value = (loadingMsgIndex.value + 1) % loadingTexts.length;
    loadingText.value = loadingTexts[loadingMsgIndex.value];
  }, 1200);
};
const stopLoadingAnimation = () => {
  if (loadingInterval) {
    clearInterval(loadingInterval);
    loadingInterval = null;
  }
};

// 모달
const openSourceModal = () => {
  isSourceModalOpen.value = true;
};
const closeSourceModal = () => {
  isSourceModalOpen.value = false;
};

// 입력 타입 선택
const selectInputType = (type) => {
  inputType.value = type;
};

// 입력 처리
const formatFileSize = (bytes) => {
  if (bytes === 0) return "0 Byte";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
};

const handleFileChange = (e) => {
  const newFile = e.target.files ? e.target.files[0] : null;
  if (!newFile) return;

  const newFileObject = {
    id: Date.now() + Math.random(),
    file: newFile,
    name: newFile.name,
    size: newFile.size,
    type: newFile.type,
  };

  file.value = newFileObject;
  if (e.target) e.target.value = null;
};

const removeFile = () => {
  file.value = null;
};

const severityColor = (severity = "Medium") => {
  switch (severity) {
    case "Critical":
      return "#ff4d4f";
    case "High":
      return "#ff8c42";
    case "Medium":
      return "#ffbf3c";
    case "Low":
      return "#4caf50";
    default:
      return "#9ca3af";
  }
};

const decorateReport = (report) => {
  if (!report) return null;
  const issues =
    (report.issues || []).map((issue) => ({
      ...issue,
      display_meta:
        issue.display_meta ||
        {
          severity_color: severityColor(issue.severity),
          highlight_lines:
            issue.start_line && issue.end_line
              ? [issue.start_line, issue.end_line]
              : [],
        },
    })) || [];
  return { ...report, issues };
};

const detectInputType = (fileName, content) => {
  const name = (fileName || "").toLowerCase();
  const text = content || "";
  if (/curl\s+/i.test(text)) return "curl_example";
  if (
    (/\.ya?ml$/.test(name) || /\.json$/.test(name)) &&
    /openapi\s*[:"]/i.test(text)
  ) {
    return "openapi";
  }
  return "code";
};

const readFileAsText = async (fileObject) => {
  if (!fileObject?.file) throw new Error("분석할 파일을 선택해 주세요.");
  return await fileObject.file.text();
};

const ensureProject = async () => {
  if (currentProject.value) return currentProject.value;
  currentProject.value = await getOrCreateDefaultProject();
  return currentProject.value;
};

const fetchReportByScanId = async (scanId) => {
  const scan = await getScanStatus(scanId);
  if (!scan) return null;
  if (scan.status === "failed") {
    throw new Error(scan.error || "스캔이 실패했습니다.");
  }
  if (!scan.reportId) return null;
  const reportResponse = await getReport(scan.reportId);
  return decorateReport(reportResponse?.report);
};

const pollScanUntilDone = async (
  scanId,
  timeoutMs = 300000,
  intervalMs = 1500
) => {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    const report = await fetchReportByScanId(scanId);
    if (report) return report;
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  // 타임아웃 직후 한 번 더 시도해보고, 여전히 없으면 에러
  const finalReport = await fetchReportByScanId(scanId);
  if (finalReport) return finalReport;
  throw new Error("스캔 대기 시간이 초과되었습니다.");
};

// 제출 처리
const handleSubmit = async () => {
  errorMessage.value = "";

  const isUploadMode = inputType.value === "upload";
  if (isUploadMode && !file.value) {
    errorMessage.value = "분석할 코드를 업로드해 주세요.";
    return;
  }
  if (!isUploadMode && message.value.trim() === "") {
    errorMessage.value = "분석할 코드를 입력해 주세요.";
    return;
  }

  isFirstInput.value = false;
  awaitingResponse.value = true;
  startLoadingAnimation();

  try {
    const project = await ensureProject();
    if (!project?.id) {
      throw new Error("프로젝트를 준비하지 못했습니다.");
    }

    let contentText = "";
    let sourceName = "";

    if (isUploadMode) {
      contentText = await readFileAsText(file.value);
      sourceName = file.value.name || "uploaded.txt";
      submittedSource.value = file.value;
    } else {
      contentText = message.value;
      sourceName = messageTitle.value.trim() || "index.js";
      const syntheticFile = new File([contentText], sourceName, {
        type: "text/plain",
      });
      submittedSource.value = {
        id: Date.now(),
        name: sourceName,
        file: syntheticFile,
        size: syntheticFile.size,
        type: syntheticFile.type,
      };
    }

    const inputTypeForApi = detectInputType(sourceName, contentText);
    const startResponse = await startScan(project.id, {
      inputType: inputTypeForApi,
      content: contentText,
      filePath: sourceName,
    });
    const scanId = startResponse?.scanId;
    if (!scanId) throw new Error("스캔 ID를 받지 못했습니다.");

    const report = await pollScanUntilDone(scanId);

    if (!report) throw new Error("리포트 데이터를 불러오지 못했습니다.");

    analysisResult.value = report;
    currentIssueIndex.value = 0;
  } catch (err) {
    console.error("분석 요청 실패", err);
    errorMessage.value =
      err?.message || "분석 요청 중 알 수 없는 오류가 발생했습니다.";
    if (!analysisResult.value) {
      isFirstInput.value = true;
    }
  } finally {
    awaitingResponse.value = false;
    stopLoadingAnimation();
  }
};

// 결과 처리
const getLanguage = (fileName) => {
  if (!fileName) return "plaintext";
  const parts = fileName.split(".");
  if (parts.length > 1) {
    const ext = parts.pop().toLowerCase();
    const map = {
      js: "javascript",
      ts: "typescript",
      jsx: "javascript",
      tsx: "typescript",
      py: "python",
      java: "java",
      kt: "kotlin",
      kts: "kotlin",
      c: "c",
      cc: "cpp",
      cpp: "cpp",
      cxx: "cpp",
      h: "cpp",
      hh: "cpp",
      hpp: "cpp",
      hxx: "cpp",
      cs: "csharp",
      go: "go",
      html: "markup",
      css: "css",
      json: "json",
      yaml: "yaml",
      vue: "markup",
      sh: "bash",
      md: "markdown",
    };
    return map[ext] || "plaintext";
  }
  return "plaintext";
};

const currentIssue = computed(() => {
  if (!analysisResult.value || !analysisResult.value.issues) return null;
  if (analysisResult.value.issues.length === 0) return null;
  return analysisResult.value.issues[currentIssueIndex.value];
});

const issueStyle = computed(() => {
  if (currentIssue.value && currentIssue.value.display_meta) {
    const color = currentIssue.value.display_meta.severity_color || "#666";
    return {
      borderColor: color,
      color: color,
    };
  }
  return {};
});

const highlightedCode = computed(() => {
  if (!currentIssue.value?.fix_code) return "";

  const fileName = submittedSource.value?.name || "code.js";
  const langAlias = getLanguage(fileName);
  const grammar = Prism.languages[langAlias];

  return grammar
    ? Prism.highlight(currentIssue.value.fix_code, grammar, langAlias)
    : Prism.util.encode(currentIssue.value.fix_code);
});

const handleCopyCode = async () => {
  if (!currentIssue.value?.fix_code) return;
  try {
    const textarea = document.createElement("textarea");
    textarea.value = currentIssue.value.fix_code;
    textarea.style.position = "fixed";
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();
    document.execCommand("copy");
    document.body.removeChild(textarea);

    isCopied.value = true;
    setTimeout(() => {
      isCopied.value = false;
    }, 2000);
  } catch (err) {
    console.error("코드 복사 실패", err);
  }
};

const nextIssue = () => {
  if (
    analysisResult.value &&
    currentIssueIndex.value < analysisResult.value.issues.length - 1
  ) {
    currentIssueIndex.value++;
  }
};
const prevIssue = () => {
  if (currentIssueIndex.value > 0) {
    currentIssueIndex.value--;
  }
};

// 페이지 이탈 경고
const hasUnsavedData = computed(() => {
  return file.value || message.value.trim() !== "";
});

onBeforeRouteLeave((to, from, next) => {
  // 분석 페이지로 이동할 때는 확인 모달 없이 바로 이동
  if (to.name === "security-analysis") {
    next();
    return;
  }

  if (hasUnsavedData.value) {
    // 추후 window.confirm 대신 커스텀 모달로 변경 가능
    const confirmed = window.confirm(
      "정말 페이지를 떠나시겠습니까?\n현재 데이터는 저장되지 않습니다."
    );
    if (confirmed) next();
    else next(false);
  } else {
    next();
  }
});

const handleBeforeUnload = (e) => {
  if (hasUnsavedData.value) {
    e.preventDefault();
    return "";
  }
};

onMounted(() => {
  window.addEventListener("beforeunload", handleBeforeUnload);
});
onBeforeUnmount(() => {
  window.removeEventListener("beforeunload", handleBeforeUnload);
  stopLoadingAnimation();
});
</script>

<template>
  <div class="flex flex-col justify-between items-center gap-6 h-screen">
    <Navbar />

    <div
      v-if="errorMessage"
      class="w-full max-w-4xl px-6"
    >
      <div
        class="border border-pink-500/70 bg-pink-500/10 text-pink-100 text-sm rounded-lg px-4 py-3"
      >
        {{ errorMessage }}
      </div>
    </div>

    <!-- 1. 입력 -->
    <main
      v-if="isFirstInput"
      class="flex flex-col grow gap-16 justify-center items-center w-full h-full px-16"
    >
      <svg
        class="size-16 sm:size-24 fill-stone-500"
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
            d="M18.3281 5.67L6.58813 17.41C6.14813 17.85 5.40813 17.79 5.04813 17.27C3.80813 15.46 3.07812 13.32 3.07812 11.12V6.73C3.07812 5.91 3.69813 4.98 4.45813 4.67L10.0281 2.39C11.2881 1.87 12.6881 1.87 13.9481 2.39L17.9981 4.04C18.6581 4.31 18.8281 5.17 18.3281 5.67Z"
          ></path>
          <path
            d="M19.27 7.04159C19.92 6.49159 20.91 6.96159 20.91 7.81159V11.1216C20.91 16.0116 17.36 20.5916 12.51 21.9316C12.18 22.0216 11.82 22.0216 11.48 21.9316C10.06 21.5316 8.74001 20.8616 7.61001 19.9816C7.13001 19.6116 7.08001 18.9116 7.50001 18.4816C9.68001 16.2516 16.06 9.75159 19.27 7.04159Z"
          ></path>
        </g>
      </svg>

      <form
        @submit.prevent="handleSubmit"
        class="flex flex-col items-end justify-center gap-4 w-fit h-fit max-w-4xl min-w-96 rounded-xl p-4"
      >
        <div class="flex justify-between items-center w-full">
          <div class="max-h-48 overflow-y-auto">
            <UploadedFile
              v-if="inputType === 'upload' && file"
              :key="file.id"
              :file="file"
              :formatFileSize="formatFileSize"
              @remove="removeFile"
            />
          </div>

          <div
            class="flex justify-center items-center shrink-0 w-fit h-fit gap-3"
          >
            <div
              class="flex rounded-xl border border-stone-500 overflow-hidden"
            >
              <!-- 업로드 모드 -->
              <div>
                <input
                  @change="selectInputType('upload')"
                  type="radio"
                  id="upload"
                  name="inputType"
                  class="peer sr-only"
                  checked
                />
                <label
                  for="upload"
                  title="코드 파일 업로드"
                  class="group flex justify-center items-center gap-2 px-2.5 py-2 text-stone-400 cursor-pointer border-r border-stone-500 peer-checked:underline peer-checked:bg-stone-800"
                >
                  <svg
                    class="size-6 fill-stone-400"
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
                        d="M20.5 10.19H17.61C15.24 10.19 13.31 8.26 13.31 5.89V3C13.31 2.45 12.86 2 12.31 2H8.07C4.99 2 2.5 4 2.5 7.57V16.43C2.5 20 4.99 22 8.07 22H15.93C19.01 22 21.5 20 21.5 16.43V11.19C21.5 10.64 21.05 10.19 20.5 10.19ZM11.53 13.53C11.38 13.68 11.19 13.75 11 13.75C10.81 13.75 10.62 13.68 10.47 13.53L9.75 12.81V17C9.75 17.41 9.41 17.75 9.00 17.75C8.59 17.75 8.25 17.41 8.25 17V12.81L7.53 13.53C7.24 13.82 6.76 13.82 6.47 13.53C6.18 13.24 6.18 12.76 6.47 12.47L8.47 10.47C8.54 10.41 8.61 10.36 8.69 10.32C8.71 10.31 8.74 10.3 8.76 10.29C8.82 10.27 8.88 10.26 8.95 10.25C8.98 10.25 9 10.25 9.03 10.25C9.11 10.25 9.19 10.27 9.27 10.3C9.28 10.3 9.28 10.3 9.29 10.3C9.37 10.33 9.45 10.39 9.51 10.45C9.52 10.46 9.53 10.46 9.53 10.47L11.53 12.47C11.82 12.76 11.82 13.24 11.53 13.53Z"
                      ></path>
                      <path
                        d="M17.4297 8.81048C18.3797 8.82048 19.6997 8.82048 20.8297 8.82048C21.3997 8.82048 21.6997 8.15048 21.2997 7.75048C19.8597 6.30048 17.2797 3.69048 15.7997 2.21048C15.3897 1.80048 14.6797 2.08048 14.6797 2.65048V6.14048C14.6797 7.60048 15.9197 8.81048 17.4297 8.81048Z"
                      ></path>
                    </g>
                  </svg>
                  <span class="text-sm group-hover:underline">Upload</span>
                </label>
              </div>
              <!-- 입력 모드 -->
              <div>
                <input
                  @change="selectInputType('write')"
                  type="radio"
                  id="write"
                  name="inputType"
                  class="peer sr-only"
                />
                <label
                  for="write"
                  title="코드 직접 입력"
                  class="group flex justify-center items-center gap-2 px-2.5 py-2 text-stone-400 cursor-pointer border-l border-stone-500 peer-checked:underline peer-checked:bg-stone-800 peer-disabled:cursor-not-allowed peer-disabled:bg-stone-700"
                >
                  <svg
                    class="size-6 fill-stone-400"
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
                        d="M21 22H3C2.59 22 2.25 21.66 2.25 21.25C2.25 20.84 2.59 20.5 3 20.5H21C21.41 20.5 21.75 20.84 21.75 21.25C21.75 21.66 21.41 22 21 22Z"
                      ></path>
                      <path
                        d="M19.0206 3.48162C17.0806 1.54162 15.1806 1.49162 13.1906 3.48162L11.9806 4.69162C11.8806 4.79162 11.8406 4.95162 11.8806 5.09162C12.6406 7.74162 14.7606 9.86162 17.4106 10.6216C17.4506 10.6316 17.4906 10.6416 17.5306 10.6416C17.6406 10.6416 17.7406 10.6016 17.8206 10.5216L19.0206 9.31162C20.0106 8.33162 20.4906 7.38162 20.4906 6.42162C20.5006 5.43162 20.0206 4.47162 19.0206 3.48162Z"
                      ></path>
                      <path
                        d="M15.6103 11.5308C15.3203 11.3908 15.0403 11.2508 14.7703 11.0908C14.5503 10.9608 14.3403 10.8208 14.1303 10.6708C13.9603 10.5608 13.7603 10.4008 13.5703 10.2408C13.5503 10.2308 13.4803 10.1708 13.4003 10.0908C13.0703 9.81078 12.7003 9.45078 12.3703 9.05078C12.3403 9.03078 12.2903 8.96078 12.2203 8.87078C12.1203 8.75078 11.9503 8.55078 11.8003 8.32078C11.6803 8.17078 11.5403 7.95078 11.4103 7.73078C11.2503 7.46078 11.1103 7.19078 10.9703 6.91078C10.9491 6.86539 10.9286 6.82022 10.9088 6.77532C10.7612 6.442 10.3265 6.34455 10.0688 6.60231L4.34032 12.3308C4.21032 12.4608 4.09032 12.7108 4.06032 12.8808L3.52032 16.7108C3.42032 17.3908 3.61032 18.0308 4.03032 18.4608C4.39032 18.8108 4.89032 19.0008 5.43032 19.0008C5.55032 19.0008 5.67032 18.9908 5.79032 18.9708L9.63032 18.4308C9.81032 18.4008 10.0603 18.2808 10.1803 18.1508L15.9016 12.4295C16.1612 12.1699 16.0633 11.7245 15.7257 11.5804C15.6877 11.5642 15.6492 11.5476 15.6103 11.5308Z"
                      ></path>
                    </g>
                  </svg>
                  <span class="text-sm group-hover:underline">Write</span>
                </label>
              </div>
            </div>

            <button
              type="submit"
              :disabled="
                (inputType === 'upload' && !file) ||
                (inputType === 'write' && message.trim() === '')
              "
              class="group flex justify-center items-center border border-stone-500 p-2 rounded-xl cursor-pointer hover:border-stone-400 disabled:cursor-not-allowed disabled:opacity-50"
              title="분석 시작"
            >
              <svg
                class="size-8 fill-stone-500 group-hover:fill-stone-400"
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
                    d="M16.1391 2.95907L7.10914 5.95907C1.03914 7.98907 1.03914 11.2991 7.10914 13.3191L9.78914 14.2091L10.6791 16.8891C12.6991 22.9591 16.0191 22.9591 18.0391 16.8891L21.0491 7.86907C22.3891 3.81907 20.1891 1.60907 16.1391 2.95907ZM16.4591 8.33907L12.6591 12.1591C12.5091 12.3091 12.3191 12.3791 12.1291 12.3791C11.9391 12.3791 11.7491 12.3091 11.5991 12.1591C11.3091 11.8691 11.3091 11.3891 11.5991 11.0991L15.3991 7.27907C15.6891 6.98907 16.1691 6.98907 16.4591 7.27907C16.7491 7.56907 16.7491 8.04907 16.4591 8.33907Z"
                  ></path>
                </g>
              </svg>
            </button>
          </div>
        </div>

        <!-- 업로드 -->
        <div
          v-if="inputType === 'upload'"
          class="flex justify-center items-center w-3xl h-64"
        >
          <input
            type="file"
            accept=".js,.ts,.jsx,.tsx,.py,.java,.kt,.kts,.c,.cc,.cpp,.cxx,.h,.hh,.hpp,.hxx,.cs,.rb,.go,.php,.html,.css,.json,.xml,.yaml,.yml,.vue,.sh,.md"
            id="file-upload"
            class="hidden"
            ref="fileInputRef"
            @change="handleFileChange"
          />
          <label
            for="file-upload"
            title="클릭해서 업로드"
            class="flex flex-col items-center justify-center gap-6 size-full border border-dashed border-pink-500 text-sm rounded-xl cursor-pointer hover:bg-stone-800/20 sm:text-base"
          >
            <svg
              class="size-12 fill-stone-500"
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
                  d="M18.6897 11.5308C18.1197 11.3808 17.4497 11.3008 16.6497 11.3008C15.5397 11.3008 15.1297 11.5708 14.5597 12.0008C14.5297 12.0208 14.4997 12.0508 14.4697 12.0808L13.5197 13.0908C12.7197 13.9308 11.2797 13.9408 10.4797 13.0808L9.52969 12.0808C9.49969 12.0508 9.46969 12.0208 9.43969 12.0008C8.86969 11.5708 8.45969 11.3008 7.34969 11.3008C6.54969 11.3008 5.87969 11.3808 5.30969 11.5308C2.92969 12.1708 2.92969 14.0608 2.92969 15.7208V16.6508C2.92969 19.1608 2.92969 22.0008 8.27969 22.0008H15.7197C19.2697 22.0008 21.0697 20.2008 21.0697 16.6508V15.7208C21.0697 14.0608 21.0697 12.1708 18.6897 11.5308ZM14.3297 18.4008H9.66969C9.28969 18.4008 8.97969 18.0908 8.97969 17.7008C8.97969 17.3108 9.28969 17.0008 9.66969 17.0008H14.3297C14.7097 17.0008 15.0197 17.3108 15.0197 17.7008C15.0197 18.0908 14.7097 18.4008 14.3297 18.4008Z"
                ></path>
                <path
                  d="M12.6684 4.3L13.3184 4.95C13.5784 5.21 14.0084 5.21 14.2684 4.95C14.5284 4.69 14.5284 4.26 14.2684 4L12.4684 2.2C12.4084 2.14 12.3284 2.09 12.2484 2.05C12.1684 2.02 12.0884 2 11.9984 2C11.9084 2 11.8284 2.02 11.7384 2.05C11.6584 2.08 11.5984 2.13 11.5384 2.18C11.5284 2.19 11.5284 2.19 11.5184 2.19L9.71844 4C9.45844 4.26 9.45844 4.69 9.71844 4.95C9.97844 5.21 10.4084 5.21 10.6684 4.95L11.3184 4.3V6H12.6684V4.3Z"
                ></path>
                <path
                  d="M19.2091 10.12C19.1691 10.1 19.1191 10.09 19.0791 10.08H19.0691C18.3591 9.89 17.5691 9.8 16.6491 9.8C15.1091 9.8 14.3791 10.25 13.7191 10.75C13.5791 10.85 13.4791 10.95 13.3791 11.05L12.4291 12.06C12.3291 12.16 12.1691 12.22 11.9991 12.22C11.9391 12.22 11.7191 12.21 11.5691 12.05L10.5891 11.02C10.4891 10.91 10.3691 10.82 10.3391 10.8C9.61906 10.25 8.88906 9.8 7.34906 9.8C6.42906 9.8 5.63906 9.89 4.91906 10.08C4.87906 10.09 4.82906 10.1 4.78906 10.12C4.79906 8.13 4.99906 6 9.20906 6H11.3291V9.08C11.3291 9.45 11.6291 9.75 11.9991 9.75C12.3691 9.75 12.6691 9.45 12.6691 9.08V6H14.7891C18.9991 6 19.1991 8.13 19.2091 10.12Z"
                ></path>
              </g>
            </svg>
            Click to Upload
          </label>
        </div>
        <!-- 입력 -->
        <div
          v-else-if="inputType === 'write'"
          class="flex flex-col justify-center items-center w-3xl h-96 rounded-xl border border-dashed border-pink-500"
        >
          <input
            v-model="messageTitle"
            name="typed-code-title"
            type="text"
            placeholder="index.js"
            title="코드 파일명"
            class="flex items-center px-4 w-full h-12 text-stone-400 text-center rounded-xl focus:outline-none caret-pink-500 placeholder:text-stone-500 hover:bg-stone-800/20 focus:bg-stone-800/20"
          />
          <textarea
            v-model="message"
            name="typed-code"
            placeholder="Write your code here..."
            class="px-4 py-2 size-full text-stone-300 rounded-xl resize-none whitespace-pre text-sm overflow-auto focus:outline-none caret-pink-500 placeholder:text-stone-500 scrollbar-thin"
          ></textarea>
        </div>
      </form>
    </main>

    <!-- 2. 로딩 -->
    <main
      v-else-if="awaitingResponse"
      class="flex flex-col justify-center items-center gap-6 h-full"
    >
      <svg
        class="size-16 sm:size-24 fill-stone-500 animate-pulse"
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
            d="M18.3281 5.67L6.58813 17.41C6.14813 17.85 5.40813 17.79 5.04813 17.27C3.80813 15.46 3.07812 13.32 3.07812 11.12V6.73C3.07812 5.91 3.69813 4.98 4.45813 4.67L10.0281 2.39C11.2881 1.87 12.6881 1.87 13.9481 2.39L17.9981 4.04C18.6581 4.31 18.8281 5.17 18.3281 5.67Z"
          ></path>
          <path
            d="M19.27 7.04159C19.92 6.49159 20.91 6.96159 20.91 7.81159V11.1216C20.91 16.0116 17.36 20.5916 12.51 21.9316C12.18 22.0216 11.82 22.0216 11.48 21.9316C10.06 21.5316 8.74001 20.8616 7.61001 19.9816C7.13001 19.6116 7.08001 18.9116 7.50001 18.4816C9.68001 16.2516 16.06 9.75159 19.27 7.04159Z"
          ></path>
        </g>
      </svg>
      <span class="text-lg animate-pulse">{{ loadingText }}</span>
    </main>

    <!-- 3. 결과 -->
    <main
      v-else-if="!awaitingResponse && !isFirstInput"
      class="flex flex-col justify-start items-center size-full gap-2 pt-4 pb-8 overflow-hidden"
    >
      <button
        @click="openSourceModal"
        class="group flex justify-center items-center px-4 py-2 rounded-xl border border-stone-500 cursor-pointer mb-4"
      >
        <svg
          class="size-5 fill-stone-500 shrink-0"
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
              d="M15.7997 2.21048C15.3897 1.80048 14.6797 2.08048 14.6797 2.65048V6.14048C14.6797 7.60048 15.9197 8.81048 17.4297 8.81048C18.3797 8.82048 19.6997 8.82048 20.8297 8.82048C21.3997 8.82048 21.6997 8.15048 21.2997 7.75048C19.8597 6.30048 17.2797 3.69048 15.7997 2.21048Z"
            ></path>
            <path
              d="M20.5 10.19H17.61C15.24 10.19 13.31 8.26 13.31 5.89V3C13.31 2.45 12.86 2 12.31 2H8.07C4.99 2 2.5 4 2.5 7.57V16.43C2.5 20 4.99 22 8.07 22H15.93C19.01 22 21.5 20 21.5 16.43V11.19C21.5 10.64 21.05 10.19 20.5 10.19ZM11.5 17.75H7.5C7.09 17.75 6.75 17.41 6.75 17C6.75 16.59 7.09 16.25 7.5 16.25H11.5C11.91 16.25 12.25 16.59 12.25 17C12.25 17.41 11.91 17.75 11.5 17.75ZM13.5 13.75H7.5C7.09 13.75 6.75 13.41 6.75 13C6.75 12.59 7.09 12.25 7.5 12.25H13.5C13.91 12.25 14.25 12.59 14.25 13C14.25 13.41 13.91 13.75 13.5 13.75Z"
            ></path>
          </g>
        </svg>
        <span
          class="text-sm pl-2 text-stone-400 group-hover:underline"
          title="원본 코드 보기"
          >Original Code</span
        >
      </button>

      <section
        class="flex justify-center items-stretch size-full gap-8 h-full px-8 min-h-0"
      >
        <!-- 왼쪽: 수정된 코드 -->
        <div
          class="group flex flex-col rounded-xl border border-stone-500 w-5/12 h-full overflow-hidden relative"
        >
          <!-- 헤더 -->
          <div
            class="flex items-center justify-between bg-stone-900 px-4 py-2 border-b border-stone-500 text-sm h-11"
          >
            <span class="text-stone-400 font-medium">수정된 코드 (Fixed)</span>
            <span
              v-if="isCopied"
              class="flex items-center gap-2 text-xs text-stone-400 uppercase cursor-default"
              title="복사 완료"
              >Copied to Clipboard</span
            >
            <span
              v-else-if="currentIssue"
              @click="handleCopyCode"
              class="text-xs text-stone-500 uppercase hover:underline group-hover:text-stone-400 cursor-pointer"
              title="클릭해서 복사"
              >Click to Copy</span
            >
          </div>

          <pre
            v-if="currentIssue"
            class="scrollbar-thin scrollbar-thumb-stone-600 scrollbar-track-stone-800 overflow-auto h-full p-4 m-0 text-sm"
          ><code v-html="highlightedCode" :class="`language-${getLanguage(submittedSource?.name)}`"></code></pre>
          <div
            v-else
            class="flex flex-col justify-center items-center gap-6 h-full text-stone-500"
          >
            <svg
              class="size-12 fill-stone-500"
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
                  d="M5.97 1H3.03C1.76 1 1 1.76 1 3.03V5.97C1 7.24 1.76 8 3.03 8H5.97C7.24 8 8 7.24 8 5.97V3.03C8 1.76 7.24 1 5.97 1ZM6.47 5.56C6.72 5.81 6.72 6.22 6.47 6.47C6.34 6.59 6.17 6.65 6.01 6.65C5.85 6.65 5.69 6.59 5.56 6.47L4.49 5.41L3.45 6.47C3.32 6.59 3.16 6.65 2.98 6.65C2.82 6.65 2.66 6.59 2.53 6.47C2.28 6.22 2.28 5.81 2.53 5.56L3.6 4.5L2.54 3.45C2.29 3.2 2.29 2.79 2.54 2.54C2.79 2.29 3.2 2.29 3.45 2.54L4.49 3.6L5.55 2.54C5.8 2.29 6.21 2.29 6.46 2.54C6.71 2.79 6.71 3.2 6.46 3.45L5.41 4.5L6.47 5.56Z"
                ></path>
                <path
                  d="M21.4995 15.8197C21.4995 15.9697 21.4495 16.1197 21.3195 16.2497C19.8695 17.7097 17.2895 20.3097 15.8095 21.7997C15.6795 21.9397 15.5095 21.9997 15.3395 21.9997C15.0095 21.9997 14.6895 21.7397 14.6895 21.3597V17.8597C14.6895 16.3997 15.9295 15.1897 17.4495 15.1897C18.3995 15.1797 19.7195 15.1797 20.8495 15.1797C21.2395 15.1797 21.4995 15.4897 21.4995 15.8197Z"
                ></path>
                <path
                  d="M21.4995 15.8197C21.4995 15.9697 21.4495 16.1197 21.3195 16.2497C19.8695 17.7097 17.2895 20.3097 15.8095 21.7997C15.6795 21.9397 15.5095 21.9997 15.3395 21.9997C15.0095 21.9997 14.6895 21.7397 14.6895 21.3597V17.8597C14.6895 16.3997 15.9295 15.1897 17.4495 15.1897C18.3995 15.1797 19.7195 15.1797 20.8495 15.1797C21.2395 15.1797 21.4995 15.4897 21.4995 15.8197Z"
                ></path>
                <path
                  d="M16.63 2H10.5C9.95 2 9.5 2.45 9.5 3V6.5C9.5 8.16 8.16 9.5 6.5 9.5H3.5C2.95 9.5 2.5 9.95 2.5 10.5V17.13C2.5 19.82 4.68 22 7.37 22H12.19C12.74 22 13.19 21.55 13.19 21V17.86C13.19 15.56 15.1 13.69 17.45 13.69C17.98 13.68 19.27 13.68 20.5 13.68C21.05 13.68 21.5 13.24 21.5 12.68V6.87C21.5 4.18 19.32 2 16.63 2ZM8.72 17.01H6.08C5.67 17.01 5.33 16.67 5.33 16.26C5.33 15.84 5.67 15.5 6.08 15.5H8.72C9.15 15.5 9.47 15.84 9.47 16.26C9.47 16.67 9.15 17.01 8.72 17.01ZM11.51 13.3H6.08C5.67 13.3 5.33 12.96 5.33 12.55C5.33 12.13 5.67 11.79 6.08 11.79H11.51C11.92 11.79 12.27 12.13 12.27 12.55C12.27 12.96 11.92 13.3 11.51 13.3Z"
                ></path>
              </g>
            </svg>
            수정된 코드가 없습니다.
          </div>
        </div>

        <!-- 오른쪽: 설명 -->
        <div
          class="flex flex-col rounded-xl border border-stone-500 w-5/12 h-full overflow-hidden"
        >
          <!-- 헤더 & 내비게이션 -->
          <div
            class="flex justify-between items-center bg-stone-900 px-4 py-2 border-b border-stone-500 h-11"
          >
            <div v-if="currentIssue" class="text-sm text-stone-400">
              발견된 취약점 {{ currentIssueIndex + 1 }} /
              {{ analysisResult.issues.length }}
            </div>
            <div v-else class="text-sm text-stone-400">발견된 취약점 0 / 0</div>
            <div v-if="currentIssue" class="flex gap-2">
              <button
                @click="prevIssue"
                :disabled="currentIssueIndex === 0"
                class="text-xs px-2 py-1 rounded border border-stone-600 hover:bg-stone-700 cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed text-stone-300"
              >
                이전
              </button>
              <button
                @click="nextIssue"
                :disabled="
                  currentIssueIndex === analysisResult.issues.length - 1
                "
                class="text-xs px-2 py-1 rounded border border-stone-600 hover:bg-stone-700 cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed text-stone-300"
              >
                다음
              </button>
            </div>
          </div>

          <!-- 내용 -->
          <div
            class="flex flex-col p-6 gap-6 overflow-y-auto scrollbar-thin scrollbar-thumb-stone-600 h-full"
          >
            <div>
              <span
                class="text-stone-500 text-xs font-bold uppercase tracking-wider"
                >Vulnerability</span
              >
              <h2
                v-if="currentIssue"
                class="text-xl font-bold text-stone-200 mt-1"
              >
                {{ currentIssue.name }}
              </h2>
              <h2 v-else class="text-xl font-bold text-stone-200 mt-1">
                취약점 미발견
              </h2>
            </div>

            <div class="flex gap-4 items-center mt-4">
              <span
                v-if="currentIssue"
                class="px-3 py-1 rounded-full text-sm font-semibold uppercase border-2 text-white"
                :style="issueStyle"
              >
                {{ currentIssue.severity }}
              </span>
              <span
                v-else
                class="px-3 py-1 rounded-full text-sm font-semibold uppercase border-2 text-green-500 border-green-500"
              >
                Safe
              </span>
              <span v-if="currentIssue" class="text-stone-400 text-sm"
                >Confidence:
                {{ (currentIssue.confidence * 100).toFixed(0) }}%</span
              >
            </div>

            <div class="border-t border-stone-700 pt-4 mt-4">
              <span
                class="text-stone-500 text-xs font-bold uppercase tracking-wider"
                >Description & Analysis</span
              >
              <p
                v-if="currentIssue"
                class="text-stone-300 mt-2 text-xs leading-relaxed whitespace-pre-wrap"
              >
                {{ currentIssue.description }}
              </p>
              <p
                v-else
                class="text-stone-300 mt-2 text-xs leading-relaxed whitespace-pre-wrap"
              >
                발견된 취약점이 적습니다. 기본 보안 점검을 계속 유지하세요.
              </p>
            </div>

            <div
              v-if="currentIssue && currentIssue.exploit_example"
              class="bg-stone-800/50 rounded-lg p-4 border border-stone-700 mt-4"
            >
              <span
                class="text-xs font-bold uppercase tracking-wider"
                :style="{
                  color:
                    currentIssue.display_meta.severity_color ||
                    'oklch(59.2% 0.249 0.584)',
                }"
                >Potential Exploit Scenario</span
              >
              <p class="text-stone-300 mt-1 text-xs leading-relaxed">
                {{ currentIssue.exploit_example }}
              </p>
            </div>
          </div>
        </div>
      </section>
    </main>
  </div>

  <UploadedFileModal
    v-if="submittedSource"
    :is-open="isSourceModalOpen"
    @close="closeSourceModal"
    :file="submittedSource"
  />
</template>

<style>
code[class*="language-"],
pre[class*="language-"] {
  background-color: oklch(21.6% 0.006 56.043) !important; /* stone-900 */
  text-shadow: none !important;
  font-family: Consolas, Monaco, "Andale Mono", "Ubuntu Mono", monospace;
  font-size: 12px;
}

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
</style>
